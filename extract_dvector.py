# import os
# import torch
# import torchaudio
# import numpy as np
# from tqdm import tqdm
# from speechbrain.inference.speaker import EncoderClassifier

# def main():
#     ref_root = "/scratch/s6295509/TSE/TIGER/tse_dataset/train-100/train_references"
#     save_root = "/scratch/s6295509/TSE/TIGER/speaker_embedding/train_embedding"
    
#     os.makedirs(save_root, exist_ok=True)
#     device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     # Load model
#     classifier = EncoderClassifier.from_hparams(
#         source="speechbrain/spkrec-ecapa-voxceleb", 
#         run_opts={"device": device}
#     )

#     spk_dirs = [d for d in os.listdir(ref_root) if os.path.isdir(os.path.join(ref_root, d))]
#     print(f"Found {len(spk_dirs)} speaker directories.")

#     for spk in tqdm(spk_dirs):
#         spk_path = os.path.join(ref_root, spk)
#         output_spk_dir = os.path.join(save_root, spk)
#         os.makedirs(output_spk_dir, exist_ok=True)

#         # [Key fix]: match .flac files, also handle case-insensitivity
#         audio_files = [f for f in os.listdir(spk_path) if f.lower().endswith('.flac')]
        
#         for f in audio_files:
#             audio_path = os.path.join(spk_path, f)
#             # Save as .npy
#             save_path = os.path.join(output_spk_dir, f.replace('.flac', '.npy').replace('.FLAC', '.npy'))
            
#             if os.path.exists(save_path): continue

#             try:
#                 # torchaudio supports loading flac directly
#                 signal, fs = torchaudio.load(audio_path)
#                 with torch.no_grad():
#                     embeddings = classifier.encode_batch(signal)
                
#                 emb_np = embeddings.squeeze().cpu().numpy()
#                 np.save(save_path, emb_np)
#             except Exception as e:
#                 print(f"Error processing {spk}/{f}: {e}")

# if __name__ == "__main__":
#     main()



# import os
# import torch
# import torchaudio
# import numpy as np
# from tqdm import tqdm
# from speechbrain.inference.speaker import EncoderClassifier

# def main():
#     # --- Modify the path here ---
#     ref_root = "/scratch/s6295509/TSE/TIGER/tse_dataset/test/test_references"
#     save_root = "/scratch/s6295509/TSE/TIGER/speaker_embedding/test_embedding"
    
#     os.makedirs(save_root, exist_ok=True)
#     device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     # Load model
#     classifier = EncoderClassifier.from_hparams(
#         source="speechbrain/spkrec-ecapa-voxceleb", 
#         run_opts={"device": device}
#     )

#     spk_dirs = [d for d in os.listdir(ref_root) if os.path.isdir(os.path.join(ref_root, d))]
#     print(f"Found {len(spk_dirs)} test speaker directories.")

#     for spk in tqdm(spk_dirs):
#         spk_path = os.path.join(ref_root, spk)
#         output_spk_dir = os.path.join(save_root, spk)
#         os.makedirs(output_spk_dir, exist_ok=True)

#         # Match .flac files
#         audio_files = [f for f in os.listdir(spk_path) if f.lower().endswith('.flac')]
        
#         for f in audio_files:
#             audio_path = os.path.join(spk_path, f)
#             save_path = os.path.join(output_spk_dir, f.replace('.flac', '.npy').replace('.FLAC', '.npy'))
            
#             if os.path.exists(save_path): continue

#             try:
#                 signal, fs = torchaudio.load(audio_path)
#                 # If the sample rate isn't 16k, resample it (just in case)
#                 if fs != 16000:
#                     resampler = torchaudio.transforms.Resample(fs, 16000)
#                     signal = resampler(signal)
                
#                 with torch.no_grad():
#                     embeddings = classifier.encode_batch(signal)
                
#                 emb_np = embeddings.squeeze().cpu().numpy()
#                 np.save(save_path, emb_np)
#             except Exception as e:
#                 print(f"Error processing {spk}/{f}: {e}")

#     print(f"Test Embeddings saved to: {save_root}")

# if __name__ == "__main__":
#     main()


import os
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from speechbrain.inference.speaker import EncoderClassifier

def main():
    # --- Validation set path ---
    ref_root = "/scratch/s6295509/TSE/TIGER/tse_dataset/dev/dev_references"
    save_root = "/scratch/s6295509/TSE/TIGER/speaker_embedding/dev_embedding"
    
    os.makedirs(save_root, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load pretrained model
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", 
        run_opts={"device": device}
    )

    spk_dirs = [d for d in os.listdir(ref_root) if os.path.isdir(os.path.join(ref_root, d))]
    print(f"Found {len(spk_dirs)} dev speaker directories.")

    for spk in tqdm(spk_dirs):
        spk_path = os.path.join(ref_root, spk)
        output_spk_dir = os.path.join(save_root, spk)
        os.makedirs(output_spk_dir, exist_ok=True)

        # Match .flac format
        audio_files = [f for f in os.listdir(spk_path) if f.lower().endswith('.flac')]
        
        for f in audio_files:
            audio_path = os.path.join(spk_path, f)
            save_path = os.path.join(output_spk_dir, f.replace('.flac', '.npy').replace('.FLAC', '.npy'))
            
            if os.path.exists(save_path): continue

            try:
                signal, fs = torchaudio.load(audio_path)
                # Resample to 16kHz
                if fs != 16000:
                    resampler = torchaudio.transforms.Resample(fs, 16000)
                    signal = resampler(signal)
                
                with torch.no_grad():
                    embeddings = classifier.encode_batch(signal)
                
                emb_np = embeddings.squeeze().cpu().numpy()
                np.save(save_path, emb_np)
            except Exception as e:
                print(f"Error processing {spk}/{f}: {e}")

    print(f"Dev Embeddings saved to: {save_root}")

if __name__ == "__main__":
    main()