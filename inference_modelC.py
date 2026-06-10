import os
import sys
import torch
import numpy as np
import torchaudio
import yaml
from tqdm import tqdm

# --- 1. Environment and path configuration ---
project_root = "/scratch/s6295509/TSE/TIGER"
sys.path.insert(0, project_root)

# Import your model class
from look2hear.models.tiger_modelC import TIGER_ModelC
from look2hear.system.audio_litmodule_modelC import AudioLightningModule_ModelC
from look2hear.metrics.wrapper import MetricsTracker

# Path definitions
MIX_DIR = "/scratch/s6295509/TSE/TIGER/tse_dataset/test/mix_clean"
S1_DIR = "/scratch/s6295509/TSE/TIGER/tse_dataset/test/s1"
EMB_DIR = "/scratch/s6295509/TSE/TIGER/tse_dataset/test/test_embedding"
CONF_PATH = "/scratch/s6295509/TSE/TIGER/configs/tiger-small_modelC.yml"
CKPT_PATH = "/scratch/s6295509/TSE/TIGER/Experiments_modelC/checkpoint/TIGER-NewMonitor-ModelC-Alpha30/epoch=127.ckpt"

# Result saving
SAVE_WAV_DIR = "/scratch/s6295509/TSE/TIGER/results/modelC_test_wavs"
CSV_SAVE_PATH = "/scratch/s6295509/TSE/TIGER/results/modelC_results.csv"

os.makedirs(SAVE_WAV_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CSV_SAVE_PATH), exist_ok=True)

def main():
    # Detect GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current device: {device}")

    # --- 2. Load config and initialize model structure ---
    with open(CONF_PATH, 'r') as f:
        config = yaml.safe_load(f)

    print("Instantiating model structure...")
    audio_model = TIGER_ModelC(
        sample_rate=config["datamodule"]["data_config"]["sample_rate"],
        **config["audionet"]["audionet_config"],
    )

    # Instantiate the LightningModule
    system = AudioLightningModule_ModelC(
        audio_model=audio_model,
        loss_func=None, 
        optimizer=None,
        train_loader=None,
        val_loader=None,
        test_loader=None,
        scheduler=None,
        config=config
    )

    # --- 3. Force-load weights ---
    print(f"Loading weights from checkpoint: {CKPT_PATH}")
    checkpoint = torch.load(CKPT_PATH, map_location="cpu")
    system.load_state_dict(checkpoint['state_dict'], strict=True)
    
    system.to(device)
    system.eval()
    system.freeze()

    # --- 4. Initialize metrics tracker ---
    tracker = MetricsTracker(save_file=CSV_SAVE_PATH)
    
    # --- 5. Batch inference loop ---
    mix_files = [f for f in os.listdir(MIX_DIR) if f.endswith('.wav')]
    print(f"Found {len(mix_files)} test audio files. Starting inference...")

    for filename in tqdm(mix_files):
        try:
            # Parse the filename to get the Speaker ID (e.g., 61-xxx -> 61)
            spk_id = filename.split('-')[0]
            
            # Build paths
            mix_path = os.path.join(MIX_DIR, filename)
            s1_path = os.path.join(S1_DIR, filename)
            spk_emb_dir = os.path.join(EMB_DIR, spk_id)
            
            # Load audio (sample rate alignment is handled in the dataloader, so just load directly here)
            mix, sr = torchaudio.load(mix_path)
            target, _ = torchaudio.load(s1_path)
            
            # Load d-vector (.npy)
            emb_file = os.listdir(spk_emb_dir)[0]
            d_vector = torch.from_numpy(np.load(os.path.join(spk_emb_dir, emb_file))).float().to(device)
            if d_vector.ndim == 1:
                d_vector = d_vector.unsqueeze(0) # [1, 192]

            # Inference
            mix_in = mix.unsqueeze(0).to(device) # [1, 1, T]
            with torch.no_grad():
                # Model C takes (mix, d_vector)
                est_target = system.audio_model(mix_in, d_vector)

            # --- 6. Compute metrics ---
            # The tracker handles computation internally and writes to CSV
            tracker(
                mix=mix.squeeze(0), 
                clean=target, 
                estimate=est_target.squeeze(0).cpu(), 
                key=filename
            )

            # Visual check: save the first 5 result audio files
            if len(tracker.all_sdrs) <= 5:
                out_name = f"modelC_est_{filename}"
                torchaudio.save(os.path.join(SAVE_WAV_DIR, out_name), est_target.squeeze(0).cpu(), sr)

        except Exception as e:
            print(f"Skipping file {filename}, reason: {e}")

    # --- 7. Final summary ---
    tracker.final()
    
    print("\n" + "="*40)
    print("[Model C Test Report]")
    print(f"Average SI-SNRi: {np.array(tracker.all_sisnrs_i).mean():.2f} dB")
    print(f"Average SDRi:    {np.array(tracker.all_sdrs_i).mean():.2f} dB")
    print(f"Detailed report path: {CSV_SAVE_PATH}")
    print(f"Sample audio path: {SAVE_WAV_DIR}")
    print("="*40)

if __name__ == "__main__":
    main()







# # s2
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

# # Import your model class
# from look2hear.models.tiger_modelC import TIGER_ModelC
# from look2hear.system.audio_litmodule_modelC import AudioLightningModule_ModelC
# from look2hear.metrics.wrapper import MetricsTracker

# # --- Path definitions (please confirm S2_DIR according to your actual setup) ---
# MIX_DIR = "/scratch/s6295509/TSE/TIGER/tse_dataset/test/mix_clean"
# S2_DIR = "/scratch/s6295509/TSE/TIGER/tse_dataset/test/s2"  # Extract the second speaker, corresponds to the s2 directory
# EMB_DIR = "/scratch/s6295509/TSE/TIGER/tse_dataset/test/test_embedding"
# CONF_PATH = "/scratch/s6295509/TSE/TIGER/configs/tiger-small_modelC.yml"
# CKPT_PATH = "/scratch/s6295509/TSE/TIGER/Experiments_modelC/checkpoint/TIGER-ModelC/epoch=0.ckpt"

# # Result saving
# SAVE_WAV_DIR = "/scratch/s6295509/TSE/TIGER/results/modelC_test_wavs_s2"
# CSV_SAVE_PATH = "/scratch/s6295509/TSE/TIGER/results/modelC_results_s2.csv"

# os.makedirs(SAVE_WAV_DIR, exist_ok=True)
# os.makedirs(os.path.dirname(CSV_SAVE_PATH), exist_ok=True)

# def main():
#     # Detect GPU
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Current device: {device}")

#     # --- 2. Load config and initialize model structure ---
#     with open(CONF_PATH, 'r') as f:
#         config = yaml.safe_load(f)

#     print("Instantiating model structure...")
#     audio_model = TIGER_ModelC(
#         sample_rate=config["datamodule"]["data_config"]["sample_rate"],
#         **config["audionet"]["audionet_config"],
#     )

#     # Instantiate the LightningModule
#     system = AudioLightningModule_ModelC(
#         audio_model=audio_model,
#         loss_func=None, 
#         optimizer=None,
#         train_loader=None,
#         val_loader=None,
#         test_loader=None,
#         scheduler=None,
#         config=config
#     )

#     # --- 3. Load weights ---
#     print(f"Loading weights from checkpoint: {CKPT_PATH}")
#     checkpoint = torch.load(CKPT_PATH, map_location="cpu")
#     system.load_state_dict(checkpoint['state_dict'], strict=True)
    
#     system.to(device)
#     system.eval()
#     system.freeze()

#     # --- 4. Initialize metrics tracker ---
#     tracker = MetricsTracker(save_file=CSV_SAVE_PATH)
    
#     # --- 5. Batch inference loop ---
#     mix_files = [f for f in os.listdir(MIX_DIR) if f.endswith('.wav')]
#     print(f"Found {len(mix_files)} test audio files. Starting extraction of the second speaker audio...")

#     for filename in tqdm(mix_files):
#         try:
#             # --- Key logic: parse the filename to get the second speaker (s2) ID ---
#             # Format: 5105-28233-0002_1089-134686-0021.wav
#             # split('_') -> ['5105-28233-0002', '1089-134686-0021.wav']
#             parts = filename.split('_')
#             if len(parts) < 2:
#                 continue
                
#             # Extract the first ID from the second segment -> 1089
#             spk_id_s2 = parts[1].split('-')[0]
            
#             # Build paths
#             mix_path = os.path.join(MIX_DIR, filename)
#             s2_path = os.path.join(S2_DIR, filename) # Clean audio control group
#             spk_emb_dir = os.path.join(EMB_DIR, spk_id_s2)
            
#             if not os.path.exists(spk_emb_dir):
#                 print(f"Warning: could not find embedding directory for Speaker {spk_id_s2}")
#                 continue

#             # Load audio
#             mix, sr = torchaudio.load(mix_path)
#             target, _ = torchaudio.load(s2_path)
            
#             # Load the first d-vector (.npy) for this speaker
#             emb_files = [f for f in os.listdir(spk_emb_dir) if f.endswith('.npy')]
#             if not emb_files:
#                 continue
#             d_vector = torch.from_numpy(np.load(os.path.join(spk_emb_dir, emb_files[0]))).float().to(device)
            
#             if d_vector.ndim == 1:
#                 d_vector = d_vector.unsqueeze(0) # [1, 192]

#             # Inference
#             mix_in = mix.unsqueeze(0).to(device) # [1, 1, T]
#             with torch.no_grad():
#                 # Pass in the mixed audio and the second speaker embedding
#                 est_target = system.audio_model(mix_in, d_vector)

#             # --- 6. Compute metrics ---
#             tracker(
#                 mix=mix.squeeze(0), 
#                 clean=target, 
#                 estimate=est_target.squeeze(0).cpu(), 
#                 key=filename
#             )

#             # Visual check: save the first 5 results
#             if len(tracker.all_sdrs) <= 5:
#                 out_name = f"modelC_s2_est_{filename}"
#                 torchaudio.save(os.path.join(SAVE_WAV_DIR, out_name), est_target.squeeze(0).cpu(), sr)

#         except Exception as e:
#             print(f"Skipping file {filename}, reason: {e}")

#     # --- 7. Final summary ---
#     tracker.final()
    
#     print("\n" + "="*40)
#     print("[Model C Test Report - Second Speaker (S2) Extraction]")
#     print(f"Average SI-SNRi: {np.array(tracker.all_sisnrs_i).mean():.2f} dB")
#     print(f"Average SDRi:    {np.array(tracker.all_sdrs_i).mean():.2f} dB")
#     print(f"Detailed report path: {CSV_SAVE_PATH}")
#     print("="*40)

# if __name__ == "__main__":
#     main()





