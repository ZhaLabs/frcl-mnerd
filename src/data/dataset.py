import os
import re
import torch
import logging
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from transformers import BertTokenizer
from torchvision import transforms
from src.config import Config

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TwitterDataset(Dataset):
    """
    Implements Sub-modules 1.1 (Text) and 1.2 (Image)
    """
    def __init__(self, data_file, image_dir, tokenizer=None, max_len=Config.MAX_SEQ_LEN):
        """
        Args:
            data_file (str): Path to the text data (e.g., train.txt or csv)
            image_dir (str): Path to the folder containing images
            tokenizer: HuggingFace BertTokenizer
            max_len: N (Max Sequence Length)
        """
        self.image_dir = image_dir
        self.max_len = max_len
        
        # Load Data (Assumes a specific format, adjust 'load_data' as needed)
        # Assuming CSV for now: [text, image_id, label]
        # If it's the standard Twitter formatting, we might need a custom parser.
        self.samples = self.load_data(data_file)
        
        self.tokenizer = tokenizer if tokenizer else BertTokenizer.from_pretrained(Config.BERT_MODEL_NAME)
        
        # Sub-Module 1.2: Image Standardization
        self.image_transform = transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ToTensor(),
            # Normalize with ImageNet stats as per spec
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def load_data(self, file_path):
        """
        Parses the dataset file. 
        Placeholder implementation assuming a tab-separated file or CSV.
        Expected columns: 'text', 'image_id'
        """
        data = []
        try:
            # Example parsing logic - REPLACE with actual format parser
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                data = df.to_dict('records')
            elif file_path.endswith('.txt'):
                # Simple parser for line-based format
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            data.append({'text': parts[0], 'image_id': parts[1]})
            logger.info(f"Loaded {len(data)} samples from {file_path}")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
        return data

    def clean_text(self, text):
        """
        Sub-Module 1.1 Step 1: Cleaning
        Remove URLs, @mentions, excessive whitespace.
        """
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'@\w+', '', text)     # Remove Mentions
        text = re.sub(r'\s+', ' ', text).strip()
        return text.lower()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        raw_text = sample.get('text', '')
        image_id = sample.get('image_id', '')
        
        # --- Sub-Module 1.1: Text Tokenizer ---
        # Step 1: Cleaning
        clean_text = self.clean_text(raw_text)
        
        # Steps 2, 3, 4, 5: Tokenization, Structuring, Padding, Mask Generation
        # BERT Tokenizer handles [CLS], [SEP], Padding, and Truncation internally
        encoding = self.tokenizer.encode_plus(
            clean_text,
            add_special_tokens=True,      # Add [CLS] and [SEP]
            max_length=self.max_len,
            padding='max_length',         # Pad to N
            truncation=True,              # Truncate if > N
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)      # [N]
        attention_mask = encoding['attention_mask'].squeeze(0) # [N]
        
        # --- Sub-Module 1.2: Image Patcher ---
        # Step 1: Resizing & Normalization
        image_path = os.path.join(self.image_dir, f"{image_id}")
        # Handle extensions if image_id doesn't have them
        if not os.path.exists(image_path):
             for ext in ['.jpg', '.png', '.jpeg']:
                 if os.path.exists(image_path + ext):
                     image_path += ext
                     break
        
        try:
            image = Image.open(image_path).convert('RGB')
            pixel_values = self.image_transform(image) # [3, 224, 224]
        except Exception:
            # Handle missing or corrupt images with a blank image
            pixel_values = torch.zeros((3, Config.IMAGE_SIZE, Config.IMAGE_SIZE))
            
        # Step 2: Patch Generation (Slicing) handled in the Model/Embedding layer usually,
        # but can be done here if strictly following spec. 
        # However, it's efficient to keep spatial structure [3, 224, 224] until the model.
        # IF we must return Raw Patch Vectors [P x D_patch] here:
        patches = self.patch_image(pixel_values) # [P, D_patch]

        return {
            'input_ids': input_ids,           # For Sub-module 1.3
            'attention_mask': attention_mask, # For Output
            'patches': patches,               # For Sub-module 1.3
            'raw_text': raw_text
        }

    def patch_image(self, pixel_values):
        """
        Sub-Module 1.2 Step 2: Patch Generation
        Slice [3, 224, 224] -> [P, D_patch]
        P = 196, D_patch = 16*16*3 = 768
        """
        c, h, w = pixel_values.shape
        p = Config.PATCH_SIZE
        
        # Unfold extracts patches
        # Input: [C, H, W] -> Unfold -> [C, D_patch, P]
        # We need to reshape carefully.
        # Using torch.nn.functional.unfold requires a batch dim, so we add/remove it.
        pixel_values = pixel_values.unsqueeze(0) # [1, 3, 224, 224]
        
        patches = torch.nn.functional.unfold(
            pixel_values, 
            kernel_size=(p, p), 
            stride=(p, p)
        ) # [1, C*p*p, L] where L is number of patches
        
        # Transpose to [1, L, C*p*p] -> [1, 196, 768]
        patches = patches.transpose(1, 2).squeeze(0) 
        return patches
