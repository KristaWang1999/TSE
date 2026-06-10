# import os
# import sys
# import torch
# import numpy as np
# import torchaudio
# import yaml
# from tqdm import tqdm

# # --- 1. Environment and path configuration ---
# project_root = "/scratch/s6295509/TSE/TIGER"
# sys.path.insert(0, project_root)

# # Import your model class and system module
# # Note: Baseline usually uses the original class without the _modelA/B suffix
# from look2hear.models.tiger import TIGER as TIGER_Baseline
# from look2hear.system.audio_litmodule import AudioLightningModule
# from look2hear.metrics.wrapper import MetricsTracker

# # Path definitions
# DATA_ROOT = "/scratch/s6295509/TSE/TIGER/dataset/Libri2Mix/wav16k/min/test"
# MIX_DIR = os.path.join(DATA_ROOT, "mix_clean")
# S1_DIR = os.path.join(DATA_ROOT, "s1")
# S2_DIR = os.path.join(DATA_ROOT, "s2")

# # Weights path
# CKPT_PATH = "/scratch/s6295509/TSE/TIGER/Experiments/checkpoint/TIGER-Libri2Mix/epoch=304.ckpt"

# # Result save path
# SAVE_WAV_DIR = "/scratch/s6295509/TSE/TIGER/results/baseline_test_wavs"
# CSV_SAVE_PATH = "/scratch/s6295509/TSE/TIGER/results/baseline_results.csv"

# os.makedirs(SAVE_WAV_DIR, exist_ok=True)
# os.makedirs(os.path.dirname(CSV_SAVE_PATH), exist_ok=True)

# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Current device: {device}")

#     # --- 2. Build a fake config dict (resolves KeyError: 'datamodule') ---
#     fake_config = {
#         "datamodule": {
#             "data_config": {
#                 "sample_rate": 16000
#             }
#         }
#     }

#     # --- 3. Instantiate model structure ---
#     # Parameters strictly correspond to the .yml config file you provided
#     print("Instantiating Baseline TIGER (SS task) structure...")
#     audio_model = TIGER_Baseline(
#         out_channels=128,
#         in_channels=256,
#         num_blocks=4,
#         upsampling_depth=5,
#         win=640,
#         stride=160,
#         num_sources=2,
#         sample_rate=16000
#     )

#     # Wrap into LightningModule
#     system = AudioLightningModule(
#         audio_model=audio_model,
#         config=fake_config
#     )

#     # --- 4. Load weights ---
#     print(f"Loading checkpoint: {CKPT_PATH}")
#     checkpoint = torch.load(CKPT_PATH, map_location="cpu")
    
#     # Support both save formats
#     state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
#     system.load_state_dict(state_dict, strict=True)
    
#     system.to(device)
#     system.eval()
#     system.freeze()

#     # --- 5. Initialize metrics tracker ---
#     # MetricsTracker automatically handles PIT (Permutation Invariant) matching for multi-source separation
#     tracker = MetricsTracker(save_file=CSV_SAVE_PATH)
    
#     mix_files = [f for f in os.listdir(MIX_DIR) if f.endswith('.wav')]
#     print(f"Found {len(mix_files)} test audio files. Starting inference...")

#     for filename in tqdm(mix_files):
#         try:
#             # Build paths
#             mix_path = os.path.join(MIX_DIR, filename)
#             s1_path = os.path.join(S1_DIR, filename)
#             s2_path = os.path.join(S2_DIR, filename)
            
#             # Load audio
#             mix, sr = torchaudio.load(mix_path)
#             s1, _ = torchaudio.load(s1_path)
#             s2, _ = torchaudio.load(s2_path)
            
#             # Concatenate the two ground truths for evaluation: [2, T]
#             targets = torch.cat([s1, s2], dim=0) 

#             # Model inference
#             mix_in = mix.unsqueeze(0).to(device) # [1, 1, T]
#             with torch.no_grad():
#                 # Pure TIGER outputs two tracks: [1, 2, T]
#                 est_sources = system.audio_model(mix_in) 

#             # --- 6. Compute metrics ---
#             tracker(
#                 mix=mix.squeeze(0), 
#                 clean=targets, 
#                 estimate=est_sources.squeeze(0).cpu(), 
#                 key=filename
#             )

#             # Save the first 5 results as samples
#             if len(tracker.all_sdrs) <= 5:
#                 for i in range(2):
#                     out_name = f"baseline_spk{i+1}_{filename}"
#                     torchaudio.save(
#                         os.path.join(SAVE_WAV_DIR, out_name), 
#                         est_sources[0, i:i+1].cpu(), 
#                         sr
#                     )

#         except Exception as e:
#             print(f"Skipping file {filename}, reason: {e}")

#     # --- 7. Final summary ---
#     tracker.final()
    
#     # Compute the mean
#     final_sisnri = np.array(tracker.all_sisnrs_i).mean()
#     final_sdri = np.array(tracker.all_sdrs_i).mean()

#     print("\n" + "="*40)
#     print("[Baseline (Pure TIGER) Final Report]")
#     print(f"Average SI-SNRi: {final_sisnri:.2f} dB")
#     print(f"Average SDRi:    {final_sdri:.2f} dB")
#     print(f"Detailed CSV path: {CSV_SAVE_PATH}")
#     print("="*40)

# if __name__ == "__main__":
#     main()







# Added PESQ

import os
import sys
import torch
import numpy as np
import torchaudio
import yaml
from tqdm import tqdm
from pesq import pesq  # Import the PESQ library

# --- 1. Environment and path configuration ---
project_root = "/scratch/s6295509/TSE/TIGER"
sys.path.insert(0, project_root)

from look2hear.models.tiger import TIGER as TIGER_Baseline
from look2hear.system.audio_litmodule import AudioLightningModule
from look2hear.metrics.wrapper import MetricsTracker

# Path definitions (please verify your dataset paths)
DATA_ROOT = "/scratch/s6295509/TSE/TIGER/dataset/Libri2Mix/wav16k/min/test"
MIX_DIR = os.path.join(DATA_ROOT, "mix_clean")
S1_DIR = os.path.join(DATA_ROOT, "s1")
S2_DIR = os.path.join(DATA_ROOT, "s2")

# Weights path (confirm your checkpoint filename)
CKPT_PATH = "/scratch/s6295509/TSE/TIGER/Experiments/checkpoint/TIGER-Libri2Mix/epoch=304.ckpt"

# Result save path
SAVE_WAV_DIR = "/scratch/s6295509/TSE/TIGER/results/baseline_test_wavs"
CSV_SAVE_PATH = "/scratch/s6295509/TSE/TIGER/results/baseline_results_with_pesq.csv"

os.makedirs(SAVE_WAV_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CSV_SAVE_PATH), exist_ok=True)

def compute_pesq_score(clean, estimate, sr=16000):
    """
    Compute the average PESQ score across the two tracks
    clean: [2, T]
    estimate: [2, T]
    """
    clean_np = clean.numpy()
    estimate_np = estimate.numpy()
    scores = []
    
    for i in range(clean_np.shape[0]):
        try:
            # Use wideband mode 'wb' for 16kHz audio
            # PESQ return values range from -0.5 to 4.5
            s = pesq(sr, clean_np[i], estimate_np[i], 'wb')
            scores.append(s)
        except Exception:
            # On silence or extreme distortion errors, default to the lowest score
            scores.append(1.0)
    return np.mean(scores)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current device: {device}")

    # --- 2. Build config ---
    fake_config = {"datamodule": {"data_config": {"sample_rate": 16000}}}

    # --- 3. Instantiate model ---
    audio_model = TIGER_Baseline(
        out_channels=128, in_channels=256, num_blocks=4,
        upsampling_depth=5, win=640, stride=160,
        num_sources=2, sample_rate=16000
    )

    system = AudioLightningModule(audio_model=audio_model, config=fake_config)

    # --- 4. Load weights ---
    print(f"Loading checkpoint: {CKPT_PATH}")
    checkpoint = torch.load(CKPT_PATH, map_location="cpu")
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    system.load_state_dict(state_dict, strict=True)
    
    system.to(device).eval()
    system.freeze()

    # --- 5. Initialize metrics ---
    tracker = MetricsTracker(save_file=CSV_SAVE_PATH)
    all_pesq = [] # Used to store PESQ scores
    
    mix_files = [f for f in os.listdir(MIX_DIR) if f.endswith('.wav')]
    print(f"Found {len(mix_files)} test audio files. Computing PESQ takes a while, please be patient...")

    # --- 6. Inference loop ---
    for filename in tqdm(mix_files):
        try:
            mix, sr = torchaudio.load(os.path.join(MIX_DIR, filename))
            s1, _ = torchaudio.load(os.path.join(S1_DIR, filename))
            s2, _ = torchaudio.load(os.path.join(S2_DIR, filename))
            targets = torch.cat([s1, s2], dim=0) 

            # Inference
            mix_in = mix.unsqueeze(0).to(device)
            with torch.no_grad():
                est_sources = system.audio_model(mix_in) 
            
            est_audio = est_sources.squeeze(0).cpu()
            target_audio = targets.cpu()

            # Compute SI-SNRi / SDRi (PIT handled automatically internally)
            tracker(
                mix=mix.squeeze(0), 
                clean=target_audio, 
                estimate=est_audio, 
                key=filename
            )

            # Compute PESQ
            current_pesq = compute_pesq_score(target_audio, est_audio, sr=16000)
            all_pesq.append(current_pesq)

            # Only save the first 5 samples
            if len(all_pesq) <= 5:
                for i in range(2):
                    torchaudio.save(
                        os.path.join(SAVE_WAV_DIR, f"baseline_spk{i+1}_{filename}"), 
                        est_audio[i:i+1], sr
                    )

        except Exception as e:
            print(f"Failed to process file {filename}: {e}")

    # --- 7. Summary report ---
    tracker.final()
    
    final_sisnri = np.array(tracker.all_sisnrs_i).mean()
    final_sdri = np.array(tracker.all_sdrs_i).mean()
    final_pesq = np.array(all_pesq).mean()

    print("\n" + "="*50)
    print("[Baseline (Pure TIGER) Final Evaluation Report]")
    print(f"Average SI-SNRi: {final_sisnri:.2f} dB  (higher is better)")
    print(f"Average SDRi:    {final_sdri:.2f} dB")
    print(f"Average PESQ:    {final_pesq:.2f}       (1.0-4.5, higher is clearer)")
    print("-" * 50)
    print(f"Detailed data saved to: {CSV_SAVE_PATH}")
    print("="*50)

if __name__ == "__main__":
    main()


# Remember to pip install pesq first