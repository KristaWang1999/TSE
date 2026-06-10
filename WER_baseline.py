# import os
# import sys
# import torch
# import numpy as np
# import torchaudio
# import whisper
# import jiwer
# from tqdm import tqdm

# # --- 1. Environment and path configuration ---
# project_root = "/scratch/s6295509/TSE/TIGER"
# sys.path.insert(0, project_root)

# from look2hear.models.tiger import TIGER as TIGER_Baseline
# from look2hear.system.audio_litmodule import AudioLightningModule

# # Path definitions
# DATA_ROOT = "/scratch/s6295509/TSE/TIGER/dataset/Libri2Mix/wav16k/min/test"
# MIX_DIR = os.path.join(DATA_ROOT, "mix_clean")
# S1_DIR = os.path.join(DATA_ROOT, "s1")
# S2_DIR = os.path.join(DATA_ROOT, "s2")
# LIBRISPEECH_ROOT = "/scratch/s6295509/speechseparation/espnet2/dataset/LibriSpeech/test-clean"

# # Weights and output
# CKPT_PATH = "/scratch/s6295509/TSE/TIGER/Experiments/checkpoint/TIGER-Libri2Mix/epoch=304.ckpt"
# SAVE_WAV_DIR = "/scratch/s6295509/TSE/TIGER/results/baseline_test_wavs"
# FINAL_REPORT_PATH = "/scratch/s6295509/TSE/TIGER/results/baseline_full_metrics.csv"

# os.makedirs(SAVE_WAV_DIR, exist_ok=True)

# # --- 2. Preprocessing logic ---
# transformation = jiwer.Compose([
#     jiwer.ToLowerCase(),
#     jiwer.RemovePunctuation(),
#     jiwer.RemoveMultipleSpaces(),
#     jiwer.Strip(),
# ])

# def get_all_transcripts(root_path):
#     trans_map = {}
#     print("Scanning LibriSpeech transcripts...")
#     for root, _, files in os.walk(root_path):
#         for f in files:
#             if f.endswith(".trans.txt"):
#                 with open(os.path.join(root, f), 'r') as f_in:
#                     for line in f_in:
#                         parts = line.strip().split(' ', 1)
#                         if len(parts) == 2: trans_map[parts[0]] = parts[1]
#     return trans_map

# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # --- 3. Load models (TIGER + Whisper) ---
#     print("Loading TIGER inference model...")
#     audio_model = TIGER_Baseline(out_channels=128, in_channels=256, num_blocks=4,
#                                  upsampling_depth=5, win=640, stride=160,
#                                  num_sources=2, sample_rate=16000)
#     system = AudioLightningModule(audio_model=audio_model, config={"datamodule": {"data_config": {"sample_rate": 16000}}})
#     checkpoint = torch.load(CKPT_PATH, map_location="cpu")
#     system.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
#     system.to(device).eval()

#     print("Loading Whisper 'base' ASR model...")
#     asr_model = whisper.load_model("base", device=device)
    
#     trans_map = get_all_transcripts(LIBRISPEECH_ROOT)
#     mix_files = [f for f in os.listdir(MIX_DIR) if f.endswith('.wav')]
    
#     results = []
    
#     # --- 4. Core loop: separate -> recognize -> compute ---
#     print(f"Starting full evaluation ({len(mix_files)} audio files)...")
#     for filename in tqdm(mix_files):
#         try:
#             # Load mixed audio
#             mix, sr = torchaudio.load(os.path.join(MIX_DIR, filename))
            
#             # Inference separation
#             with torch.no_grad():
#                 est_sources = system.audio_model(mix.unsqueeze(0).to(device)) # [1, 2, T]
            
#             # Parse IDs (assumes filename format: ID1_ID2.wav)
#             ids = filename.replace('.wav', '').split('_')
            
#             for i in range(2): # Process two speakers
#                 spk_id = ids[i]
#                 if spk_id not in trans_map: continue
                
#                 # Extract single-speaker audio and convert to numpy (required by Whisper)
#                 est_audio = est_sources[0, i].cpu().numpy()
                
#                 # Save audio (for later listening)
#                 out_name = f"baseline_spk{i+1}_{filename}"
#                 torchaudio.save(os.path.join(SAVE_WAV_DIR, out_name), 
#                                 torch.from_numpy(est_audio).unsqueeze(0), sr)
                
#                 # ASR recognition (pass the in-memory numpy array directly for speed)
#                 asr_res = asr_model.transcribe(est_audio, language="en", fp16=(device=="cuda"))
#                 hyp = transformation(asr_res["text"])
#                 ref = transformation(trans_map[spk_id])
                
#                 # Compute WER
#                 current_wer = jiwer.wer(ref, hyp)
                
#                 results.append({
#                     "mixture": filename,
#                     "spk": f"spk{i+1}",
#                     "wer": current_wer,
#                     "ref": ref,
#                     "hyp": hyp
#                 })

#         except Exception as e:
#             print(f"Skipping {filename}, error: {e}")

#     # --- 5. Summary ---
#     import pandas as pd
#     df = pd.DataFrame(results)
#     df.to_csv(FINAL_REPORT_PATH, index=False)
#     print("\n" + "="*50)
#     print(f"Evaluation complete! Average WER: {df['wer'].mean():.2%}")
#     print(f"Detailed report and separated audio saved to: {os.path.dirname(FINAL_REPORT_PATH)}")
#     print("="*50)

# if __name__ == "__main__":
#     main()


# WER 22.75% 
# import os
# import whisper
# from jiwer import wer
# import re

# # --- Path configuration ---
# hyp_audio_dir = "/scratch/s6295509/TSE/TIGER/results/baseline_test_wavs"
# ref_s1_path = "/scratch/s6295509/TSE/TIGER/test_ref_s1.txt"
# ref_s2_path = "/scratch/s6295509/TSE/TIGER/test_ref_s2.txt"
# output_report = "pi_wer_report.txt"

# def load_references(file_path):
#     refs = {}
#     with open(file_path, 'r') as f:
#         for line in f:
#             parts = line.strip().split(' ', 1)
#             if len(parts) > 1:
#                 refs[parts[0]] = parts[1].upper()
#     return refs

# def clean_text(text):
#     text = text.upper()
#     return re.sub(r'[^\w\s]', '', text)

# def main():
#     print("Loading Whisper model...")
#     model = whisper.load_model("medium")

#     ref_s1_dict = load_references(ref_s1_path)
#     ref_s2_dict = load_references(ref_s2_path)

#     # 1. Scan the folder, grouping by ID (e.g., 61-70968-0000_8455-210777-0012)
#     all_files = os.listdir(hyp_audio_dir)
#     mix_ids = set()
#     for f in all_files:
#         match = re.search(r"baseline_spk\d_(.+)\.wav", f)
#         if match:
#             mix_ids.add(match.group(1))

#     results = []
#     total_pi_wer = 0

#     print(f"Found {len(mix_ids)} mixture audio groups, starting pairwise computation...")

#     for mix_id in sorted(list(mix_ids)):
#         # Build the two audio paths for this group
#         path_s1 = os.path.join(hyp_audio_dir, f"baseline_spk1_{mix_id}.wav")
#         path_s2 = os.path.join(hyp_audio_dir, f"baseline_spk2_{mix_id}.wav")

#         if not os.path.exists(path_s1) or not os.path.exists(path_s2):
#             print(f"Skipping: {mix_id} is missing its separated audio pair")
#             continue

#         # Get reference text
#         r1 = ref_s1_dict.get(mix_id)
#         r2 = ref_s2_dict.get(mix_id)
#         if not r1 or not r2: continue

#         # Whisper transcription
#         h1 = clean_text(model.transcribe(path_s1, language="en")['text'])
#         h2 = clean_text(model.transcribe(path_s2, language="en")['text'])

#         # --- Permutation Invariant (PIT) computation ---
#         # Combination A: h1-r1, h2-r2
#         wer_a1 = wer(r1, h1)
#         wer_a2 = wer(r2, h2)
#         avg_wer_a = (wer_a1 + wer_a2) / 2

#         # Combination B: h1-r2, h2-r1
#         wer_b1 = wer(r2, h1)
#         wer_b2 = wer(r1, h2)
#         avg_wer_b = (wer_b1 + wer_b2) / 2

#         # Pick the best
#         if avg_wer_a <= avg_wer_b:
#             best_wer = avg_wer_a
#             swapped = False
#             final_h1, final_h2 = h1, h2
#         else:
#             best_wer = avg_wer_b
#             swapped = True
#             final_h1, final_h2 = h2, h1 # Write out the aligned results when recording

#         results.append({
#             "id": mix_id,
#             "wer": best_wer,
#             "swapped": swapped,
#             "r1": r1, "h1": h1,
#             "r2": r2, "h2": h2
#         })
#         total_pi_wer += best_wer
#         print(f"ID: {mix_id} | WER: {best_wer:.4f} | Swapped: {swapped}")

#     # 2. Generate report
#     with open(output_report, 'w') as f:
#         f.write("Permutation-Invariant (PI) WER Report\n" + "="*50 + "\n")
#         for res in results:
#             f.write(f"Mix ID: {res['id']}\n")
#             f.write(f"Swapped Channel: {res['swapped']}\n")
#             f.write(f"SPK1 Ref: {res['r1']}\n")
#             f.write(f"SPK1 Hyp: {res['h1']}\n")
#             f.write(f"SPK2 Ref: {res['r2']}\n")
#             f.write(f"SPK2 Hyp: {res['h2']}\n")
#             f.write(f"Avg WER: {res['wer']:.4f}\n")
#             f.write("-" * 30 + "\n")
        
#         final_avg = total_pi_wer / len(results) if results else 0
#         f.write(f"\nFinal Global Average PI-WER: {final_avg:.4f}\n")

#     print(f"Done! Final PI-WER: {final_avg:.4f}")

# if __name__ == "__main__":
#     main()




# # new large
# import os
# import re
# import whisper
# from jiwer import wer
# from whisper.normalizers import EnglishTextNormalizer

# # --- Path configuration (please make sure these paths match your Hábrók directory) ---
# hyp_audio_dir = "/scratch/s6295509/TSE/TIGER/results/baseline_test_wavs"
# ref_s1_path = "/scratch/s6295509/TSE/TIGER/test_ref_s1.txt"
# ref_s2_path = "/scratch/s6295509/TSE/TIGER/test_ref_s2.txt"
# output_report = "pi_wer_report_largeV3_final.txt"

# # Initialize the standard English text normalizer: handles numbers, abbreviations, punctuation
# normalizer = EnglishTextNormalizer()

# def load_references(file_path):
#     """Load reference text and store it keyed by ID"""
#     refs = {}
#     if not os.path.exists(file_path):
#         print(f"Error: reference file not found {file_path}")
#         return refs
#     with open(file_path, 'r') as f:
#         for line in f:
#             parts = line.strip().split(' ', 1)
#             if len(parts) > 1:
#                 refs[parts[0]] = parts[1]
#     return refs

# def clean_text_refined(text):
#     """
#     Deep cleaning logic:
#     1. Use the official Whisper Normalizer (10 -> ten, I'm -> I am)
#     2. Convert uniformly to uppercase
#     """
#     if not text:
#         return ""
#     # The normalizer outputs lowercase; convert to uppercase uniformly for alignment
#     return normalizer(text).upper()

# def main():
#     # Thoroughly clear environment variables that might interfere with Torch
#     if "PYTHONPATH" in os.environ:
#         del os.environ["PYTHONPATH"]

#     print("Loading Whisper large-v3 model (uses ~10GB of VRAM)...")
#     # large-v3 performs best on V100/A100 environments
#     model = whisper.load_model("large-v3")

#     print("Reading reference text...")
#     ref_s1_dict = load_references(ref_s1_path)
#     ref_s2_dict = load_references(ref_s2_path)

#     # Scan the audio folder
#     all_files = os.listdir(hyp_audio_dir)
#     mix_ids = set()
#     for f in all_files:
#         # Match the Mix ID in the filename (e.g., baseline_spk1_ID.wav)
#         match = re.search(r"baseline_spk\d_(.+)\.wav", f)
#         if match:
#             mix_ids.add(match.group(1))

#     results = []
#     total_pi_wer = 0
#     valid_count = 0

#     print(f"Found {len(mix_ids)} audio groups, starting PI-WER evaluation (PIT mode)...")

#     for mix_id in sorted(list(mix_ids)):
#         path_s1 = os.path.join(hyp_audio_dir, f"baseline_spk1_{mix_id}.wav")
#         path_s2 = os.path.join(hyp_audio_dir, f"baseline_spk2_{mix_id}.wav")

#         if not os.path.exists(path_s1) or not os.path.exists(path_s2):
#             print(f"Skipping: {mix_id} (incomplete audio)")
#             continue

#         # Get reference text
#         r1_raw = ref_s1_dict.get(mix_id)
#         r2_raw = ref_s2_dict.get(mix_id)
#         if not r1_raw or not r2_raw:
#             continue

#         # Whisper transcription (language set to English, disable prompts for speed)
#         res1 = model.transcribe(path_s1, language="en", fp16=True)
#         res2 = model.transcribe(path_s2, language="en", fp16=True)
        
#         # --- Core improvement: bidirectional deep cleaning ---
#         h1 = clean_text_refined(res1['text'])
#         h2 = clean_text_refined(res2['text'])
#         r1 = clean_text_refined(r1_raw)
#         r2 = clean_text_refined(r2_raw)

#         # --- Permutation Invariant (PIT) computation ---
#         # Combination A: 1-1, 2-2
#         wer_a = (wer(r1, h1) + wer(r2, h2)) / 2
#         # Combination B: 1-2, 2-1
#         wer_b = (wer(r1, h2) + wer(r2, h1)) / 2

#         if wer_a <= wer_b:
#             best_wer = wer_a
#             swapped = False
#             # Keep alignment when recording: H1 corresponds to R1, H2 to R2
#             final_h1, final_h2 = h1, h2
#         else:
#             best_wer = wer_b
#             swapped = True
#             # If swapped, record the transcription results swapped accordingly
#             final_h1, final_h2 = h2, h1

#         results.append({
#             "id": mix_id,
#             "wer": best_wer,
#             "swapped": swapped,
#             "r1": r1, "h1": final_h1,
#             "r2": r2, "h2": final_h2
#         })
        
#         total_pi_wer += best_wer
#         valid_count += 1
#         print(f"[{valid_count}/{len(mix_ids)}] ID: {mix_id} | WER: {best_wer:.4f} | Swapped: {swapped}")

#     # --- Generate report ---
#     with open(output_report, 'w') as f:
#         f.write("PI-WER Evaluation Report (Optimized)\n")
#         f.write(f"Model: Whisper Large-v3 | Normalizer: EnglishTextNormalizer\n")
#         f.write("="*70 + "\n")
        
#         for res in results:
#             f.write(f"ID: {res['id']}\n")
#             f.write(f"Channel Swapped: {res['swapped']}\n")
#             f.write(f"Ref 1: {res['r1']}\n")
#             f.write(f"Hyp 1: {res['h1']}\n")
#             f.write(f"Ref 2: {res['r2']}\n")
#             f.write(f"Hyp 2: {res['h2']}\n")
#             f.write(f"WER: {res['wer']:.4f}\n")
#             f.write("-" * 50 + "\n")
        
#         final_avg = total_pi_wer / valid_count if valid_count > 0 else 0
#         f.write(f"\nFinal Global Average PI-WER: {final_avg:.4f}\n")

#     print(f"\nEvaluation complete! Final PI-WER: {final_avg:.4f}")
#     print(f"Detailed report saved to: {output_report}")

# if __name__ == "__main__":
#     main()



# wer
# import os
# import re
# import torch
# import whisper
# from jiwer import wer
# from whisper.normalizers import EnglishTextNormalizer
# from tqdm import tqdm

# # --- Path configuration ---
# hyp_audio_dir  = "/scratch/s6295509/TSE/TIGER/results/baseline_test_wavs"
# ref_s1_dir     = "/scratch/s6295509/TSE/TIGER/tse_dataset/test/s1"
# ref_s2_dir     = "/scratch/s6295509/TSE/TIGER/tse_dataset/test/s2"
# output_report  = "/scratch/s6295509/TSE/TIGER/results/pi_wer_baseline_largeV3.txt"

# os.makedirs(os.path.dirname(output_report), exist_ok=True)

# normalizer = EnglishTextNormalizer()

# def clean_text(text):
#     if not text:
#         return ""
#     return normalizer(text).upper()

# def transcribe(model, path):
#     """Transcribe an audio file and return the cleaned text"""
#     result = model.transcribe(path, language="en", fp16=torch.cuda.is_available())
#     return clean_text(result["text"])

# def main():
#     if "PYTHONPATH" in os.environ:
#         del os.environ["PYTHONPATH"]

#     print("Loading Whisper large-v3 model...")
#     model = whisper.load_model("large-v3")

#     # Scan hypothesis audio and collect all mix_ids
#     mix_ids = set()
#     for f in os.listdir(hyp_audio_dir):
#         match = re.search(r"baseline_spk\d_(.+)\.wav", f)
#         if match:
#             mix_ids.add(match.group(1))
#     mix_ids = sorted(list(mix_ids))
#     print(f"Found {len(mix_ids)} audio groups, starting PI-WER evaluation...")

#     results = []
#     total_pi_wer = 0.0
#     valid_count = 0

#     for mix_id in tqdm(mix_ids):
#         # --- Hypothesis audio paths ---
#         path_h1 = os.path.join(hyp_audio_dir, f"baseline_spk1_{mix_id}.wav")
#         path_h2 = os.path.join(hyp_audio_dir, f"baseline_spk2_{mix_id}.wav")

#         # --- Reference audio paths (use s1/s2 wav directly to avoid min/max text mismatch) ---
#         path_r1 = os.path.join(ref_s1_dir, f"{mix_id}.wav")
#         path_r2 = os.path.join(ref_s2_dir, f"{mix_id}.wav")

#         # Check whether the files exist
#         missing = [p for p in [path_h1, path_h2, path_r1, path_r2]
#                    if not os.path.exists(p)]
#         if missing:
#             print(f"Skipping {mix_id}, missing files: {missing}")
#             continue

#         # --- Use Whisper to transcribe reference audio (actual content after min truncation) ---
#         r1 = transcribe(model, path_r1)
#         r2 = transcribe(model, path_r2)

#         # --- Use Whisper to transcribe hypothesis audio ---
#         h1 = transcribe(model, path_h1)
#         h2 = transcribe(model, path_h2)

#         # Skip empty transcriptions
#         if not r1 or not r2 or not h1 or not h2:
#             print(f"Skipping {mix_id}, empty transcription result")
#             continue

#         # --- PI-WER: enumerate both permutations and take the best ---
#         wer_a = (wer(r1, h1) + wer(r2, h2)) / 2   # no swap
#         wer_b = (wer(r1, h2) + wer(r2, h1)) / 2   # swapped

#         if wer_a <= wer_b:
#             best_wer, swapped = wer_a, False
#             final_h1, final_h2 = h1, h2
#         else:
#             best_wer, swapped = wer_b, True
#             final_h1, final_h2 = h2, h1  # swap so h and r align

#         results.append({
#             "id": mix_id,
#             "wer": best_wer,
#             "swapped": swapped,
#             "r1": r1, "h1": final_h1,
#             "r2": r2, "h2": final_h2,
#         })

#         total_pi_wer += best_wer
#         valid_count += 1

#     # --- Generate report ---
#     final_avg = total_pi_wer / valid_count if valid_count > 0 else 0.0

#     with open(output_report, "w") as f:
#         f.write("PI-WER Evaluation Report\n")
#         f.write("Model: Whisper Large-v3 | Ref: Whisper transcription of clean s1/s2\n")
#         f.write("=" * 70 + "\n")

#         for res in results:
#             f.write(f"ID: {res['id']}\n")
#             f.write(f"Channel Swapped: {res['swapped']}\n")
#             f.write(f"Ref 1: {res['r1']}\n")
#             f.write(f"Hyp 1: {res['h1']}\n")
#             f.write(f"Ref 2: {res['r2']}\n")
#             f.write(f"Hyp 2: {res['h2']}\n")
#             f.write(f"WER: {res['wer']:.4f}\n")
#             f.write("-" * 50 + "\n")

#         f.write(f"\nTotal evaluated: {valid_count}\n")
#         f.write(f"Final Global Average PI-WER: {final_avg:.4f}\n")

#     print(f"\nEvaluation complete! Evaluated {valid_count} items")
#     print(f"Final PI-WER: {final_avg:.4f}")
#     print(f"Report saved to: {output_report}")

# if __name__ == "__main__":
#     main()




import os
import re
import numpy as np
import soundfile as sf
from pesq import pesq
from tqdm import tqdm

# --- Path configuration (consistent with the WER script) ---
hyp_audio_dir = "/scratch/s6295509/TSE/TIGER/results/baseline_test_wavs"
ref_s1_dir    = "/scratch/s6295509/TSE/TIGER/tse_dataset/test/s1"
ref_s2_dir    = "/scratch/s6295509/TSE/TIGER/tse_dataset/test/s2"
output_csv    = "/scratch/s6295509/TSE/TIGER/results/pi_pesq_baseline.csv"
output_report = "/scratch/s6295509/TSE/TIGER/results/pi_pesq_baseline_report.txt"

os.makedirs(os.path.dirname(output_csv), exist_ok=True)

FS = 16000  # 16kHz wideband

def load_wav(path):
    audio, sr = sf.read(path)
    if sr != FS:
        raise ValueError(f"Expected {FS} Hz, got {sr} Hz: {path}")
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio.astype(np.float32)

def compute_pesq(ref, hyp):
    """Wideband PESQ, returns MOS-LQO score (range -0.5 to 4.5)."""
    # Trim/pad to same length
    min_len = min(len(ref), len(hyp))
    return pesq(FS, ref[:min_len], hyp[:min_len], 'wb')

def main():
    # Scan mix_ids
    mix_ids = set()
    for f in os.listdir(hyp_audio_dir):
        match = re.search(r"baseline_spk\d_(.+)\.wav", f)
        if match:
            mix_ids.add(match.group(1))
    mix_ids = sorted(mix_ids)
    print(f"Found {len(mix_ids)} audio groups, starting PI-PESQ evaluation...")

    results = []
    total_pesq = 0.0
    valid_count = 0

    for mix_id in tqdm(mix_ids):
        path_h1 = os.path.join(hyp_audio_dir, f"baseline_spk1_{mix_id}.wav")
        path_h2 = os.path.join(hyp_audio_dir, f"baseline_spk2_{mix_id}.wav")
        path_r1 = os.path.join(ref_s1_dir,    f"{mix_id}.wav")
        path_r2 = os.path.join(ref_s2_dir,    f"{mix_id}.wav")

        missing = [p for p in [path_h1, path_h2, path_r1, path_r2]
                   if not os.path.exists(p)]
        if missing:
            print(f"Skipping {mix_id}, missing: {missing}")
            continue

        try:
            h1 = load_wav(path_h1)
            h2 = load_wav(path_h2)
            r1 = load_wav(path_r1)
            r2 = load_wav(path_r2)

            # PI-PESQ: take the higher average score across both permutations
            pesq_a = (compute_pesq(r1, h1) + compute_pesq(r2, h2)) / 2  # no swap
            pesq_b = (compute_pesq(r1, h2) + compute_pesq(r2, h1)) / 2  # swapped

            if pesq_a >= pesq_b:
                best_pesq, swapped = pesq_a, False
                pesq_spk1 = compute_pesq(r1, h1)
                pesq_spk2 = compute_pesq(r2, h2)
            else:
                best_pesq, swapped = pesq_b, True
                pesq_spk1 = compute_pesq(r1, h2)
                pesq_spk2 = compute_pesq(r2, h1)

            results.append({
                "snt_id":    f"{mix_id}.wav",
                "pesq":      round(best_pesq, 4),
                "pesq_spk1": round(pesq_spk1, 4),
                "pesq_spk2": round(pesq_spk2, 4),
                "swapped":   swapped,
            })
            total_pesq += best_pesq
            valid_count += 1

        except Exception as e:
            print(f"Error {mix_id}: {e}")
            continue

    final_avg = total_pesq / valid_count if valid_count > 0 else 0.0

    # Save CSV
    import csv
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["snt_id", "pesq", "pesq_spk1", "pesq_spk2", "swapped"])
        writer.writeheader()
        writer.writerows(results)

    # Save text report
    with open(output_report, "w") as f:
        f.write("PI-PESQ Evaluation Report (Wideband)\n")
        f.write("=" * 50 + "\n")
        for r in results:
            f.write(f"ID: {r['snt_id']}\n")
            f.write(f"Channel Swapped: {r['swapped']}\n")
            f.write(f"PESQ Spk1: {r['pesq_spk1']}\n")
            f.write(f"PESQ Spk2: {r['pesq_spk2']}\n")
            f.write(f"PI-PESQ:   {r['pesq']}\n")
            f.write("-" * 30 + "\n")
        f.write(f"\nTotal evaluated: {valid_count}\n")
        f.write(f"Final Average PI-PESQ: {final_avg:.4f}\n")

    print(f"\nEvaluation complete! Total {valid_count} items")
    print(f"Average PI-PESQ: {final_avg:.4f}")
    print(f"CSV saved: {output_csv}")

if __name__ == "__main__":
    main()
