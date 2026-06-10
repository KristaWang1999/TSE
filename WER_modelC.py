import os
import sys
import torch
import numpy as np
import torchaudio
import yaml
import whisper
from jiwer import wer as compute_wer
from whisper.normalizers import EnglishTextNormalizer
from tqdm import tqdm
import pandas as pd
from pesq import pesq as pesq_score

# --- Environment configuration ---
project_root = "/scratch/s6295509/TSE/TIGER"
sys.path.insert(0, project_root)

from look2hear.models.tiger_modelC import TIGER_ModelC
from look2hear.system.audio_litmodule_modelC import AudioLightningModule_ModelC
from look2hear.metrics.wrapper import MetricsTracker

# --- Path configuration ---
MIX_DIR       = "/scratch/s6295509/TSE/TIGER/tse_dataset/test/mix_clean"
S1_DIR        = "/scratch/s6295509/TSE/TIGER/tse_dataset/test/s1"
EMB_DIR       = "/scratch/s6295509/TSE/TIGER/tse_dataset/test/test_embedding"
CONF_PATH     = "/scratch/s6295509/TSE/TIGER/configs/tiger-small_modelC.yml"
CKPT_PATH     = "/scratch/s6295509/TSE/TIGER/Experiments_modelC/checkpoint/TIGER-NewMonitor-ModelC-Alpha30/epoch=127.ckpt"
SAVE_WAV_DIR  = "/scratch/s6295509/TSE/TIGER/results/modelC_test_wavs"
CSV_SAVE_PATH = "/scratch/s6295509/TSE/TIGER/results/modelC_full_metrics.csv"
REPORT_PATH   = "/scratch/s6295509/TSE/TIGER/results/modelC_wer_report.txt"

os.makedirs(SAVE_WAV_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CSV_SAVE_PATH), exist_ok=True)

SAMPLE_RATE = 16000
normalizer  = EnglishTextNormalizer()


def _align(a, b):
    n = min(len(a), len(b))
    return a[:n], b[:n]

def compute_pesq(estimate, target):
    e, t = _align(estimate.flatten(), target.flatten())
    try:
        return pesq_score(SAMPLE_RATE, t, e, 'wb')
    except Exception:
        return float('nan')

def clean_text(text):
    return normalizer(text).upper() if text else ""

def transcribe(model, audio_np):
    result = model.transcribe(
        audio_np.astype(np.float32),
        language="en",
        fp16=torch.cuda.is_available()
    )
    return clean_text(result["text"])


def main():
    if "PYTHONPATH" in os.environ:
        del os.environ["PYTHONPATH"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    with open(CONF_PATH, 'r') as f:
        config = yaml.safe_load(f)

    audio_model = TIGER_ModelC(
        sample_rate=config["datamodule"]["data_config"]["sample_rate"],
        **config["audionet"]["audionet_config"],
    )
    system = AudioLightningModule_ModelC(
        audio_model=audio_model,
        loss_func=None, optimizer=None,
        train_loader=None, val_loader=None, test_loader=None,
        scheduler=None, config=config
    )
    checkpoint = torch.load(CKPT_PATH, map_location="cpu")
    system.load_state_dict(checkpoint['state_dict'], strict=True)
    system.to(device).eval()
    system.freeze()
    print("TIGER Model C loaded")

    print("Loading Whisper large-v3 model...")
    asr_model = whisper.load_model("large-v3")
    print("Whisper loaded")

    tracker = MetricsTracker(save_file=CSV_SAVE_PATH)

    mix_files = sorted([f for f in os.listdir(MIX_DIR) if f.endswith('.wav')])
    print(f"Found {len(mix_files)} test audio files, starting inference and evaluation...")

    pesq_list = []
    wer_list  = []
    ref_list  = []
    hyp_list  = []
    key_list  = []

    for filename in tqdm(mix_files):
        try:
            spk_id      = filename.split('-')[0]
            mix_path    = os.path.join(MIX_DIR, filename)
            s1_path     = os.path.join(S1_DIR, filename)
            spk_emb_dir = os.path.join(EMB_DIR, spk_id)

            mix,    sr = torchaudio.load(mix_path)
            target, _  = torchaudio.load(s1_path)
            if sr != SAMPLE_RATE:
                rs = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
                mix = rs(mix);  target = rs(target)
                sr  = SAMPLE_RATE

            emb_file = os.listdir(spk_emb_dir)[0]
            d_vector = torch.from_numpy(
                np.load(os.path.join(spk_emb_dir, emb_file))
            ).float().to(device)
            if d_vector.ndim == 1:
                d_vector = d_vector.unsqueeze(0)

            mix_in = mix.unsqueeze(0).to(device)
            with torch.no_grad():
                est_target = system.audio_model(mix_in, d_vector)

            est_cpu = est_target.squeeze(0).cpu()

            torchaudio.save(
                os.path.join(SAVE_WAV_DIR, f"modelC_est_{filename}"),
                est_cpu, sr
            )

            tracker(
                mix=mix.squeeze(0),
                clean=target,
                estimate=est_cpu,
                key=filename
            )

            est_np    = est_cpu.squeeze().numpy()
            target_np = target.squeeze().numpy()
            pesq_val  = compute_pesq(est_np, target_np)

            ref_text = transcribe(asr_model, target_np)
            hyp_text = transcribe(asr_model, est_np)
            wer_val  = compute_wer(ref_text, hyp_text) if (ref_text and hyp_text) else float('nan')

            pesq_list.append(pesq_val)
            wer_list.append(wer_val)
            ref_list.append(ref_text)
            hyp_list.append(hyp_text)
            key_list.append(filename)

        except Exception as e:
            print(f"Skipping {filename}, error: {e}")

    tracker.final()
    avg_sisnri = np.array(tracker.all_sisnrs_i).mean()
    avg_sdri   = np.array(tracker.all_sdrs_i).mean()
    avg_pesq   = np.nanmean(pesq_list)
    avg_wer    = np.nanmean(wer_list)

    try:
        df = pd.read_csv(CSV_SAVE_PATH)
        df_extra = pd.DataFrame({
            "snt_id":   key_list,
            "pesq":     [round(v, 4) if not np.isnan(v) else "nan" for v in pesq_list],
            "wer":      [round(v, 4) if not np.isnan(v) else "nan" for v in wer_list],
            "ref_text": ref_list,
            "hyp_text": hyp_list,
        })
        df = pd.merge(df, df_extra, on="snt_id", how="left")
        df.to_csv(CSV_SAVE_PATH, index=False)
    except Exception as e:
        print(f"CSV merge failed, saving separately: {e}")
        pd.DataFrame({
            "snt_id": key_list, "pesq": pesq_list,
            "wer": wer_list, "ref_text": ref_list, "hyp_text": hyp_list,
        }).to_csv(CSV_SAVE_PATH.replace(".csv", "_pesq_wer.csv"), index=False)

    with open(REPORT_PATH, "w") as f:
        f.write("Model C Evaluation Report\n")
        f.write("SI-SNRi/SDRi: MetricsTracker (BSS eval) | PESQ: wb | WER: Whisper large-v3\n")
        f.write("=" * 70 + "\n")
        for i, fname in enumerate(key_list):
            f.write(f"ID: {fname}\n")
            f.write(f"PESQ: {pesq_list[i]}\n")
            f.write(f"Ref: {ref_list[i]}\n")
            f.write(f"Hyp: {hyp_list[i]}\n")
            f.write(f"WER: {wer_list[i]}\n")
            f.write("-" * 50 + "\n")
        f.write(f"\nTotal evaluated: {len(key_list)}\n")
        f.write(f"Average SI-SNRi: {avg_sisnri:.4f} dB\n")
        f.write(f"Average SDRi:    {avg_sdri:.4f} dB\n")
        f.write(f"Average PESQ:    {avg_pesq:.4f}\n")
        f.write(f"Average WER:     {avg_wer:.4f}\n")

    print("\n" + "=" * 50)
    print("[Model C Evaluation Report]")
    print(f"Average SI-SNRi: {avg_sisnri:.2f} dB")
    print(f"Average SDRi:    {avg_sdri:.2f} dB")
    print(f"Average PESQ:    {avg_pesq:.4f}")
    print(f"Average WER:     {avg_wer:.2%}")
    print(f"CSV report:     {CSV_SAVE_PATH}")
    print(f"Text report:    {REPORT_PATH}")
    print(f"Inference audio: {SAVE_WAV_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()
