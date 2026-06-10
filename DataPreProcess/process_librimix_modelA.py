import argparse
import json
import os
import soundfile as sf
from tqdm import tqdm
import glob

def preprocess_tse(in_data_dir, out_dir):
    """TSE JSON generation script tailored to your dataset structure"""
    # Iterate over the three data splits
    for data_type in ["train-100", "dev", "test"]:
        print(f"Processing: {data_type}")
        
        # 1. Locate paths
        base_path = os.path.join(in_data_dir, data_type)
        mix_dir = os.path.join(base_path, "mix_clean")
        s1_dir = os.path.join(base_path, "s1")
        
        # Adapt folder name: train -> train_embedding, dev -> dev_embedding
        emb_folder = "train_embedding" if "train" in data_type else f"{data_type.split('-')[0]}_embedding"
        emb_root = os.path.join(base_path, emb_folder)
        
        file_infos = []
        # Get all mixture audio files
        wav_list = sorted([f for f in os.listdir(mix_dir) if f.endswith(".wav")])
        
        for wav_file in tqdm(wav_list):
            # A. Absolute paths of the mix and target (s1) audio
            mix_path = os.path.abspath(os.path.join(mix_dir, wav_file))
            target_path = os.path.abspath(os.path.join(s1_dir, wav_file))
            
            # B. Parse the speaker ID
            # Example: "19-198-0001_27-123349-0024.wav"
            # Step 1: get "19-198-0001"
            s1_full_id = wav_file.split('_')[0] 
            # Step 2: get "19" (matches the folder name)
            target_spk_id = s1_full_id.split('-')[0]
            
            # C. Locate the d-vector path
            spk_emb_dir = os.path.join(emb_root, target_spk_id)
            # Search this directory for all .npy files
            dvec_files = sorted(glob.glob(os.path.join(spk_emb_dir, "*.npy")))
            
            if not dvec_files:
                # Safety check in case some speakers are missing data
                continue
            
            # Note: train-100 has two npy files, we default to the first
            # For dev/test there is only one, so taking the first is correct too
            selected_dvec = os.path.abspath(dvec_files[0])
            
            # D. Get audio length (used by the DataLoader)
            with sf.SoundFile(mix_path) as f:
                length = len(f)
            
            # Write to dict
            file_infos.append({
                "mix": mix_path,
                "target": target_path,
                "dvector": selected_dvec,
                "length": length
            })
            
        # 2. Save JSON
        output_path = os.path.join(out_dir, data_type)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            
        with open(os.path.join(output_path, "data.json"), "w") as f:
            json.dump(file_infos, f, indent=4)
            
    print("\nAll JSON files have been generated!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, default="/scratch/s6295509/TSE/TIGER/tse_dataset")
    parser.add_argument("--out_dir", type=str, default="/scratch/s6295509/TSE/TIGER/tse_dataset")
    args = parser.parse_args()
    
    preprocess_tse(args.in_dir, args.out_dir)