import torch
from torch.utils.data import DataLoader
from src.config import Config
from src.data.dataset import TwitterDataset
from src.modules.embeddings import MultimodalPreprocessorModule
from tqdm import tqdm
import os
import argparse

def save_embeddings(dataset_name, split):
    print(f"=== Processing {dataset_name} [{split}] ===")
    
    # 1. Construct Paths
    # Assuming structure: data/NER_data/{dataset_name}_data/{dataset_name}/{split}.txt
    # Adjust this based on your actual folder structure if different
    base_dir = os.path.join(Config.DATA_DIR, f"{dataset_name}_data") # e.g. twitter15_data
    
    # Handle potentially different naming conventions if needed
    # For now assuming standard: twitter2015/train.txt inside twitter15_data folder?
    # Actually, let's verify the structure. 
    # Current twitter15 structure: data/NER_data/twitter15_data/twitter2015/train.txt
    
    # Let's try to be flexible or strictly follow the pattern found in demo.py
    # demo.py used: os.path.join(base_dir, "twitter2015", "train.txt") where base_dir was twitter15_data
    
    # Mapping for subfolder names
    subfolder_map = {
        "twitter15": "twitter2015",
        "twitter17": "twitter2017" 
    }
    
    subname = subfolder_map.get(dataset_name, dataset_name)
    
    data_file = os.path.join(base_dir, subname, f"{split}.txt")
    image_dir = os.path.join(base_dir, f"{subname}_images")
    
    output_dir = os.path.join("data", "processed", dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{split}.pt")
    
    if not os.path.exists(data_file):
        print(f"[ERROR] Data file not found: {data_file}")
        return

    # 2. Load Data
    print(f"[1] Loading Dataset from {data_file}...")
    dataset = TwitterDataset(data_file=data_file, image_dir=image_dir)
    print(f"    Size: {len(dataset)}")
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # 3. Model
    print("[2] Initializing Model...")
    model = MultimodalPreprocessorModule()
    model.to(Config.DEVICE)
    model.eval()
    
    # 4. Storage Info
    # We want to save: ImageID, Text Emb, Visual Emb, Labels (aligned)
    # Note: T_Emb shape is [B, 128, 768], V_Emb is [B, 196, 768]
    # This can get large. 4000 samples * (128+196) * 768 * 4 bytes ~= 4GB? 
    # Let's check: 4000 * 324 * 768 * 4 / 1024^3 ≈ 3.7 GB. 
    # Might be better to save in chunks or just one big file if RAM allows.
    
    all_data = {
        "image_ids": [],
        "t_emb": [],
        "v_emb": [],
        "label_ids": [],
        "attention_mask": []
    }
    
    print(f"[3] Processing...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(Config.DEVICE)
            patches = batch['patches'].to(Config.DEVICE)
            
            # Forward
            t_emb, v_emb = model(input_ids, patches)
            
            # Move to CPU to save RAM
            all_data['t_emb'].append(t_emb.cpu())
            all_data['v_emb'].append(v_emb.cpu())
            all_data['label_ids'].append(batch['label_ids'])
            all_data['attention_mask'].append(batch['attention_mask'])
            all_data['image_ids'].extend(batch['image_id'])
            
    # Concatenate
    print("[4] Saving to disk...")
    final_data = {
        "t_emb": torch.cat(all_data['t_emb'], dim=0),
        "v_emb": torch.cat(all_data['v_emb'], dim=0),
        "label_ids": torch.cat(all_data['label_ids'], dim=0),
        "attention_mask": torch.cat(all_data['attention_mask'], dim=0),
        "image_ids": all_data['image_ids']
    }
    
    torch.save(final_data, output_path)
    print(f"[SUCCESS] Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="twitter15", help="twitter15 or twitter17")
    parser.add_argument("--split", type=str, default="train", help="train, test, or valid")
    args = parser.parse_args()
    
    save_embeddings(args.dataset, args.split)
