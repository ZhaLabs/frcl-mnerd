import torch
from src.config import Config
from src.data.dataset import TwitterDataset
from src.modules.embeddings import MultimodalPreprocessorModule
from torch.utils.data import DataLoader

from tqdm import tqdm

def demo():
    print("=== Module I: Full Dataset Preprocessing & Embedding Verification ===")
    
    # 1. Setup Data Paths
    import os
    base_dir = os.path.join(Config.DATA_DIR, "twitter15_data")
    train_file = os.path.join(base_dir, "twitter2015", "train.txt")
    image_dir = os.path.join(base_dir, "twitter2015_images")

    print("[1] Initializing Dataset...")
    if not os.path.exists(train_file):
        print(f"[ERROR] Data file not found at: {train_file}")
        return

    dataset = TwitterDataset(data_file=train_file, image_dir=image_dir)
    print(f"    Dataset Size: {len(dataset)}")
    
    # 2. Setup DataLoader
    # Increased batch size for faster processing
    batch_size = 16
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # 3. Setup Embedding Module
    print("[2] Initializing Embedding Module...")
    model = MultimodalPreprocessorModule()
    model.to(Config.DEVICE)
    model.eval() # Inference mode
    
    # 4. Run Loop
    print(f"[3] Processing Entire Dataset on {Config.DEVICE}...")
    
    total_samples = 0
    augmented_masks = 0
    augmented_patches = 0
    
    # Use tqdm for progress bar
    progress_bar = tqdm(dataloader, desc="Preprocessing", unit="batch")
    
    with torch.no_grad():
        for i, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(Config.DEVICE)
            attention_mask = batch['attention_mask'].to(Config.DEVICE)
            patches = batch['patches'].to(Config.DEVICE)
            
            # Verify Augmentation/Missing Data Stats
            mask_token_id = dataset.tokenizer.mask_token_id
            augmented_masks += (input_ids == mask_token_id).sum().item()
            
            # Check for zeroed patches (failed image loads or augmentations)
            # A patch is fully zero if all dimensions are 0
            # [B, P, D] -> Check D dim
            is_zero_patch = (patches == 0).all(dim=2) # [B, P]
            # Check if *all* patches in an image are zero (indicates missing image)
            is_missing_image = is_zero_patch.all(dim=1) # [B]
            augmented_patches += is_missing_image.sum().item()
            
            # Forward Pass
            t_emb, v_emb = model(input_ids, patches)
            
            # Basic Shape Check (only on first batch to avoid console spam)
            if i == 0:
                print(f"\n[Check] First Batch Shapes:")
                print(f" - Input IDs: {input_ids.shape}")
                print(f" - T_Emb: {t_emb.shape}")
                print(f" - V_Emb: {v_emb.shape}")
            
            total_samples += input_ids.size(0)
            
    print("\n=== Processing Complete ===")
    print(f"Total Samples Processed: {total_samples}")
    print(f"Total Missing/Failed Images: {augmented_patches} ({(augmented_patches/total_samples)*100:.2f}%)")
    print("\n[SUCCESS] The entire dataset was passed through Module I without errors.")

if __name__ == "__main__":
    demo()
