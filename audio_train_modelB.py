import os
import sys


project_root = "/scratch/s6295509/TSE/TIGER"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 2. Virtual environment path (kept last as a fallback)
env_path = "/scratch/s6295509/env_tiger_tse/lib/python3.10/site-packages"
if env_path not in sys.path:
    sys.path.append(env_path)

import look2hear
# 3. Print check (this time the path should no longer be None)
print(f"--- Environment check ---")
print(f"Code root directory: {project_root}")
try:
    print(f"look2hear path: {look2hear.__file__}")
except AttributeError:
    print(f"look2hear path: could not get __file__, but import succeeded")
print(f"--------------------")

import torch
from torch import Tensor
import argparse
import json
import look2hear.datas
import look2hear.models
import look2hear.system
import look2hear.losses
import look2hear.metrics
import look2hear.utils
from look2hear.system import make_optimizer
from dataclasses import dataclass
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import *
from rich.console import Console
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from rich import print, reconfigure
from collections.abc import MutableMapping
from look2hear.utils import print_only, MyRichProgressBar, RichProgressBarTheme

import warnings

warnings.filterwarnings("ignore")

# import wandb
# wandb.login()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--conf_dir",
    default="local/conf.yml",
    help="Full path to save best validation model",
)

parser.add_argument(
    "--pretrain_path", 
    default=None, 
    help="Path to the checkpoint to load weights from"
)

parser.add_argument(
    "--model_path",
    default=None,
    help="Full path to the checkpoint to resume training from (preserves epoch)",
)

def main(config, model_path=None, pretrain_path=None):

    # This way, even if __init__.py wasn't changed, this will precisely load the class from tiger_modelB.py
    from look2hear.models.tiger_modelB import TIGER_ModelB
    from look2hear.datas.Libri2Mix16_modelB import Libri2MixModule_ModelB as DataModule
    from look2hear.system.audio_litmodule_modelB import AudioLightningModule_ModelB



    print_only(
        "Instantiating datamodule <{}>".format(config["datamodule"]["data_name"])
    )

    datamodule = DataModule(
        **config["datamodule"]["data_config"]
    )
    datamodule.setup()

    train_loader, val_loader, test_loader = datamodule.make_loader
    
    # Define model and optimizer
    print_only(f"Instantiating AudioNet <{config['audionet']['audionet_name']}>")
    
    model = TIGER_ModelB(
        sample_rate=config["datamodule"]["data_config"]["sample_rate"],
        **config["audionet"]["audionet_config"],
    )
    # import pdb; pdb.set_trace()
    print_only("Instantiating Optimizer <{}>".format(config["optimizer"]["optim_name"]))
    optimizer = make_optimizer(model.parameters(), **config["optimizer"])

    # Define scheduler
    scheduler = None
    if config["scheduler"]["sche_name"]:
        print_only(
            "Instantiating Scheduler <{}>".format(config["scheduler"]["sche_name"])
        )
        if config["scheduler"]["sche_name"] != "DPTNetScheduler":
            scheduler = getattr(torch.optim.lr_scheduler, config["scheduler"]["sche_name"])(
                optimizer=optimizer, **config["scheduler"]["sche_config"]
            )
        else:
            scheduler = {
                "scheduler": getattr(look2hear.system.schedulers, config["scheduler"]["sche_name"])(
                    optimizer, len(train_loader) // config["datamodule"]["data_config"]["batch_size"], 64
                ),
                "interval": "step",
            }

    config["exp"]["exp_name"] = config["exp"]["exp_name"] + "-Alpha30"


    # Just after instantiating, save the args. Easy loading in the future.
    config["main_args"]["exp_dir"] = os.path.join(
        os.getcwd(), "Experiments_modelB", "checkpoint", config["exp"]["exp_name"]
    )
    exp_dir = config["main_args"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(config, outfile)

    # Define Loss function.
    print_only(
        "Instantiating Loss, Train <{}>, Val <{}>".format(
            config["loss"]["train"]["sdr_type"], config["loss"]["val"]["sdr_type"]
        )
    )
    loss_func = {
        "train": getattr(look2hear.losses, config["loss"]["train"]["loss_func"])(
            getattr(look2hear.losses, config["loss"]["train"]["sdr_type"]),
            **config["loss"]["train"]["config"],
        ),
        "val": getattr(look2hear.losses, config["loss"]["val"]["loss_func"])(
            getattr(look2hear.losses, config["loss"]["val"]["sdr_type"]),
            **config["loss"]["val"]["config"],
        ),
    }

    print_only("Instantiating System <Model B FiLM Version>")
    system = AudioLightningModule_ModelB(
        audio_model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        scheduler=scheduler,
        config=config,
    )

    p_path = pretrain_path or config.get("pretrain_path")
    
    # if p_path:
    #     print_only(f"Loading weights from pretrain_path as starting point: {p_path}")
    #     checkpoint = torch.load(p_path, map_location="cpu")

    #     if "state_dict" in checkpoint:
    #         state_dict = {k.replace("audio_model.", ""): v for k, v in checkpoint["state_dict"].items() if k.startswith("audio_model.")}
    #         system.audio_model.load_state_dict(state_dict)
    #     else:
    #         system.audio_model.load_state_dict(checkpoint)
        
    #     print_only(f"Weights loaded successfully! Training will start from a fresh Epoch 0, saved into the {config['exp']['exp_name']} folder.")

    if pretrain_path is not None and model_path is None: # Only manually load pretrained weights when not resuming training
        print_only(f"Manually loading weights from checkpoint: {pretrain_path}")
        checkpoint = torch.load(pretrain_path, map_location="cpu")
        system.load_state_dict(checkpoint["state_dict"])
        print_only(f"Weights loaded successfully! Training will start from a fresh Epoch 0.")
    elif model_path is not None:
        print_only(f"Detected model_path, resuming training from checkpoint: {model_path}")

    # Define callbacks
    print_only("Instantiating ModelCheckpoint")
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir)
    checkpoint = ModelCheckpoint(
        checkpoint_dir,
        filename="{epoch}",
        monitor="val/si_snri/dataloader_idx_0",
        mode="max",
        save_top_k=5,
        verbose=True,
        save_last=True,
    )
    callbacks.append(checkpoint)

    if config["training"]["early_stop"]:
        print_only("Instantiating EarlyStopping")
        callbacks.append(EarlyStopping(**config["training"]["early_stop"]))
    callbacks.append(MyRichProgressBar(theme=RichProgressBarTheme()))

    # Don't ask GPU if they are not available.
    gpus = config["training"]["gpus"] if torch.cuda.is_available() else None
    distributed_backend = "cuda" if torch.cuda.is_available() else None

    # default logger used by trainer
    logger_dir = os.path.join(os.getcwd(), "Experiments_modelB", "tensorboard_logs")
    os.makedirs(os.path.join(logger_dir, config["exp"]["exp_name"]), exist_ok=True)
    # comet_logger = TensorBoardLogger(logger_dir, name=config["exp"]["exp_name"])
    comet_logger = WandbLogger(
            name=config["exp"]["exp_name"], 
            save_dir=os.path.join(logger_dir, config["exp"]["exp_name"]), 
            project="Real-work-dataset",
            # offline=True
    )

    trainer = pl.Trainer(
        max_epochs=config["training"]["epochs"],
        callbacks=callbacks,
        default_root_dir=exp_dir,
        devices=gpus,
        accelerator=distributed_backend,
        strategy="auto",
        limit_train_batches=1.0,  # Useful for fast experiment
        gradient_clip_val=5.0,
        logger=comet_logger,
        sync_batchnorm=False,
        # precision="bf16-mixed",
        # num_sanity_val_steps=0,
        # sync_batchnorm=True,
        # fast_dev_run=True,
    )
    trainer.fit(system, ckpt_path=model_path)
    print_only("Finished Training")
    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    to_save = system.audio_model.serialize()
    torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))


if __name__ == "__main__":
    import yaml
    from look2hear.utils.parser_utils import (
        prepare_parser_from_dict,
        parse_args_as_dict,
    )

    args = parser.parse_args()
    with open(args.conf_dir) as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)

    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    
    # --- Core fix: use getattr to avoid AttributeError ---
    m_path = getattr(plain_args, "model_path", None)
    p_path = getattr(plain_args, "pretrain_path", None)
    
    # Pass into the main function
    main(arg_dic, model_path=m_path, pretrain_path=p_path)
