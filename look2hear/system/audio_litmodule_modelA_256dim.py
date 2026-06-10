import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections.abc import MutableMapping
from speechbrain.augment.time_domain import SpeedPerturb
from speechbrain.inference.speaker import EncoderClassifier
import torch.nn.functional as F

def flatten_dict(d, parent_key="", sep="_"):
    """Flattens a dictionary into a single-level dictionary while preserving
    parent keys. Taken from
    `SO <https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys>`_

    Args:
        d (MutableMapping): Dictionary to be flattened.
        parent_key (str): String to use as a prefix to all subsequent keys.
        sep (str): String to use as a separator between two key levels.

    Returns:
        dict: Single-level dictionary, flattened.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class AudioLightningModule_ModelA(pl.LightningModule):
    def __init__(
        self,
        audio_model=None,
        video_model=None,
        optimizer=None,
        loss_func=None,
        train_loader=None,
        val_loader=None,
        test_loader=None,
        scheduler=None,
        config=None,
    ):
        super().__init__()

        print("ModelA initialized (with Speaker Loss)")

        self.audio_model = audio_model
        self.video_model = video_model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scheduler = scheduler
        self.config = {} if config is None else config
        
        
        # Initialize speaker encoder (ECAPA-TDNN)
        self.spk_encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            run_opts={"device": "cuda"}
        )
        self.spk_encoder.eval()
        for param in self.spk_encoder.parameters():
            param.requires_grad = False

        # Speed Aug
        self.speedperturb = SpeedPerturb(
            self.config["datamodule"]["data_config"]["sample_rate"],
            speeds=[95, 100, 105]
            
        )

       
        
        # Save lightning"s AttributeDict under self.hparams
        self.default_monitor = "val/si_snri/dataloader_idx_0"
        self.save_hyperparameters(self.config_to_hparams(self.config))
        # self.print(self.audio_model)
        self.validation_step_outputs = []
        self.test_step_outputs = []
        

    def forward(self, wav, dvec):
        """Applies forward pass of the model.

        """
        return self.audio_model(wav, dvec)


    def training_step(self, batch, batch_nb):
            mixtures, targets, dvec = batch

            est_sources = self(mixtures, dvec)
            preds = est_sources.squeeze(1)
            
            loss_sisnr = self.loss_func["train"](preds, targets.squeeze(1))

            # --- 3. Speaker consistency loss (amplitude normalization) ---
            self.spk_encoder.to(self.device)
            
            # Peak normalization: prevents the model from reducing amplitude to evade the speaker penalty
            with torch.no_grad():
                max_vals = torch.max(torch.abs(preds), dim=1, keepdim=True)[0]
            preds_for_spk = preds / (max_vals + 1e-8)
            
            est_embeddings = self.spk_encoder.encode_batch(preds_for_spk)
            est_dvec = est_embeddings.squeeze(1)
            
            cos_sim = F.cosine_similarity(est_dvec, dvec, dim=-1)
            loss_spk = (1 - cos_sim).mean()

            alpha = 2.0 
            total_loss = loss_sisnr + alpha * loss_spk

            self.log("train_loss/total", total_loss, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("train_loss/sisnr", loss_sisnr, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("train_loss/spk_sim", loss_spk, on_epoch=True, prog_bar=True, sync_dist=True) 
            self.log("train/cos_sim_raw", cos_sim.mean(), on_epoch=True, sync_dist=True)
            
            return total_loss

        

    def validation_step(self, batch, batch_nb, dataloader_idx=0):
        mixtures, targets, dvec = batch
        
        est_sources = self(mixtures, dvec)
        preds = est_sources.squeeze(1)
        
        # 2. Current negative SI-SNR
        current_neg_sisnr = self.loss_func["val"](preds, targets.squeeze(1))

        # 3. Baseline input SI-SNR
        with torch.no_grad():
            initial_neg_sisnr = self.loss_func["val"](mixtures.squeeze(1), targets.squeeze(1))
        
        # 4. SI-SNRi improvement
        si_snri = initial_neg_sisnr - current_neg_sisnr

        # 5. Speaker consistency metric (amplitude normalization)
        with torch.no_grad():
            self.spk_encoder.to(self.device)
            max_vals = torch.max(torch.abs(preds), dim=1, keepdim=True)[0]
            preds_norm = preds / (max_vals + 1e-8)
            
            est_embeddings = self.spk_encoder.encode_batch(preds_norm)
            est_dvec = est_embeddings.squeeze(1)
            val_cos_sim = F.cosine_similarity(est_dvec, dvec, dim=-1).mean()
            val_loss_spk = 1 - val_cos_sim

        alpha = 2.0
        val_total_loss = current_neg_sisnr + alpha * val_loss_spk

        if dataloader_idx == 0:
            self.log("val_loss", val_total_loss, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("val/si_snri", si_snri, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val/spk_sim_loss", val_loss_spk, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val/cos_sim", val_cos_sim, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("val/sisnr_loss", current_neg_sisnr, on_epoch=True, sync_dist=True)
            
            self.validation_step_outputs.append(si_snri)
        else:
            self.log("test/si_snri", si_snri, on_epoch=True, prog_bar=True, sync_dist=True)
            self.test_step_outputs.append(si_snri)
            
        return val_total_loss

    def on_validation_epoch_end(self):
        if self.validation_step_outputs:
            avg_val_snri = torch.stack(self.validation_step_outputs).mean()
            val_snri_all = torch.mean(self.all_gather(avg_val_snri))
            
            self.log("lr", self.optimizer.param_groups[0]["lr"], on_epoch=True, prog_bar=True, sync_dist=True)
            
            self.logger.experiment.log({
                "learning_rate": self.optimizer.param_groups[0]["lr"], 
                "val/si_snri_epoch": val_snri_all, 
                "epoch": self.current_epoch
            })
            self.validation_step_outputs.clear() 

        if self.test_step_outputs:
            avg_test_snri = torch.stack(self.test_step_outputs).mean()
            test_snri_all = torch.mean(self.all_gather(avg_test_snri))
            
            self.logger.experiment.log({
                "test/si_snri_epoch": test_snri_all, 
                "epoch": self.current_epoch
            })
            self.test_step_outputs.clear()

    def configure_optimizers(self):
        """Initialize optimizers, batch-wise and epoch-wise schedulers."""
        if self.scheduler is None:
            return self.optimizer

        if not isinstance(self.scheduler, (list, tuple)):
            self.scheduler = [self.scheduler]  # support multiple schedulers

        epoch_schedulers = []
        for sched in self.scheduler:
            if not isinstance(sched, dict):
                if isinstance(sched, ReduceLROnPlateau):
                    sched = {"scheduler": sched, "monitor": self.default_monitor}
                epoch_schedulers.append(sched)
            else:
                sched.setdefault("monitor", self.default_monitor)
                sched.setdefault("frequency", 1)
                # Backward compat
                if sched["interval"] == "batch":
                    sched["interval"] = "step"
                assert sched["interval"] in [
                    "epoch",
                    "step",
                ], "Scheduler interval should be either step or epoch"
                epoch_schedulers.append(sched)
        return [self.optimizer], epoch_schedulers    
    
    def train_dataloader(self):
        """Training dataloader"""
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader"""
        return [self.val_loader, self.test_loader]

    def on_save_checkpoint(self, checkpoint):
        """Overwrite if you want to save more things in the checkpoint."""
        checkpoint["training_config"] = self.config
        return checkpoint

    @staticmethod
    def config_to_hparams(dic):
        """Sanitizes the config dict to be handled correctly by torch
        SummaryWriter. It flatten the config dict, converts ``None`` to
        ``"None"`` and any list and tuple into torch.Tensors.

        Args:
            dic (dict): Dictionary to be transformed.

        Returns:
            dict: Transformed dictionary.
        """
        dic = flatten_dict(dic)
        for k, v in dic.items():
            if v is None:
                dic[k] = str(v)
            elif isinstance(v, (list, tuple)):
                dic[k] = torch.tensor(v)
        return dic
