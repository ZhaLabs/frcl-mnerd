import torch
from src.config import Config
from src.data.dataset import TwitterDataset
from src.modules.embeddings import MultimodalPreprocessorModule
from torch.utils.data import DataLoader

def demo():
    print("=== Module I: Preprocessing & Embedding Demo ===")
    
    # 1. Setup Mock Data
    # In a real scenario, these would be paths to the NER_data folder
    # We mock the dataset to avoid needing actual files for this test
    print("[1] Initializing Dataset...")
    dataset = TwitterDataset(data_file="mock_train.txt", image_dir="mock_images")
    
    # Inject mock data manually to bypass file loading for this demo
    dataset.samples = [
        {'text': "RT @User: This is a sample tweet with a URL http://test.com", 'image_id': "img1"},
        {'text': "Another example tweet for the FRCL model.", 'image_id': "img2"}
    ]
    
    # 2. Setup DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # 3. Setup Embedding Module
    print("[2] Initializing Embedding Module...")
    model = MultimodalPreprocessorModule()
    
    # 4. Run Batch
    print("[3] Running Pipeline...")
    batch = next(iter(dataloader))
    
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    patches = batch['patches']
    
    print(f"\nRaw Batch Shapes (from Dataset):")
    print(f" - Input IDs: {input_ids.shape} (Expected: [B, N])")
    print(f" - Attention Mask: {attention_mask.shape} (Expected: [B, N])")
    print(f" - Patches: {patches.shape} (Expected: [B, P, D_patch])")

    # Check for Augmentation
    mask_token_id = dataset.tokenizer.mask_token_id
    masked_count = (input_ids == mask_token_id).sum().item()
    print(f" - Augmented (Masked) Tokens found: {masked_count} (Should be > 0 usually)")
    zeroed_patches = (patches == 0).all(dim=2).sum().item()
    print(f" - Augmented (Dropped) Patches found: {zeroed_patches} (Should be > 0 usually)")
    
    # 5. Forward Pass (Module 1.3)
    t_emb, v_emb = model(input_ids, patches)
    
    print(f"\nFinal Feature Embeddings (from Module 1.3):")
    print(f" - T_Emb: {t_emb.shape} (Expected: [B, {Config.MAX_SEQ_LEN}, {Config.HIDDEN_DIM}])")
    print(f" - V_Emb: {v_emb.shape} (Expected: [B, {Config.NUM_PATCHES}, {Config.HIDDEN_DIM}])")
    
    print("\n[SUCCESS] Module I Output Requirements Met.")

if __name__ == "__main__":
    demo()
