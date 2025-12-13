import os
import torch
import logging
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
    Implements loading for Twitter-15 NER Dataset.
    Format assumptions:
    IMGID: <id>
    word1 label1
    word2 label2
    ...
    (blank line)
    """
    def __init__(self, data_file, image_dir, tokenizer=None, max_len=Config.MAX_SEQ_LEN, is_training=True):
        self.image_dir = image_dir
        self.max_len = max_len
        self.is_training = is_training
        
        # Load Data
        self.samples = self.load_data(data_file)
        
        self.tokenizer = tokenizer if tokenizer else BertTokenizer.from_pretrained(Config.BERT_MODEL_NAME)
        
        # Sub-Module 1.2: Image Standardization
        self.image_transform = transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ToTensor(),
            # Normalize with ImageNet stats
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def load_data(self, file_path):
        data = []
        if not os.path.exists(file_path):
            logger.warning(f"Data file not found: {file_path}")
            return data

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        current_id = None
        current_words = []
        current_labels = []

        for line in lines:
            line = line.strip()
            
            # Start of a new tweet/image block
            if line.startswith("IMGID:"):
                # If we have a previous block accumulating, store it
                # Note: standard format usually has blank line at end of tweet, 
                # but IMGID can also serve as separator check.
                if current_id is not None and current_words:
                    data.append({
                        'image_id': current_id,
                        'words': current_words,
                        'labels': current_labels
                    })
                
                # Reset for new block
                current_id = line.split(":", 1)[1].strip()
                current_words = []
                current_labels = []
                continue
            
            # Empty line usually signals end of sentence
            if not line:
                if current_id is not None and current_words:
                    data.append({
                        'image_id': current_id,
                        'words': current_words,
                        'labels': current_labels
                    })
                    current_id = None
                    current_words = []
                    current_labels = []
                continue

            # Word Label line
            parts = line.split()
            if len(parts) >= 2:
                # Assuming last part is label, rest is word (though usually single word)
                label = parts[-1]
                word = " ".join(parts[:-1]) 
                current_words.append(word)
                current_labels.append(label)
        
        # Catch last one if no blank line at end
        if current_id is not None and current_words:
            data.append({
                'image_id': current_id,
                'words': current_words,
                'labels': current_labels
            })

        logger.info(f"Loaded {len(data)} samples from {file_path}")
        return data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        words = sample['words']
        labels = sample['labels']
        image_id = sample['image_id']
        
        # --- Tokenization & Label Alignment ---
        input_ids = []
        label_ids = []
        
        # Add [CLS]
        input_ids.append(self.tokenizer.cls_token_id)
        label_ids.append(-100) # Ignore [CLS]
        
        for word, label in zip(words, labels):
            word_tokens = self.tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [self.tokenizer.unk_token] # Fallback
            
            word_ids = self.tokenizer.convert_tokens_to_ids(word_tokens)
            
            # Map label to ID
            l_id = Config.LABEL2ID.get(label, Config.LABEL2ID["O"])
            
            input_ids.extend(word_ids)
            # Only first subword gets the label, others get -100
            label_ids.append(l_id)
            if len(word_ids) > 1:
                label_ids.extend([-100] * (len(word_ids) - 1))
        
        # Output Truncation (saving room for [SEP])
        if len(input_ids) > self.max_len - 1:
            input_ids = input_ids[:self.max_len - 1]
            label_ids = label_ids[:self.max_len - 1]
            
        # Add [SEP]
        input_ids.append(self.tokenizer.sep_token_id)
        label_ids.append(-100)
        
        # Padding
        padding_length = self.max_len - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            label_ids = label_ids + [-100] * padding_length # Ignore padding for loss
            attention_mask = [1] * (len(input_ids) - padding_length) + [0] * padding_length
        else:
            attention_mask = [1] * len(input_ids)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        label_ids = torch.tensor(label_ids, dtype=torch.long)

        # --- Sub-Module 1.2: Image Patcher ---
        image_path = os.path.join(self.image_dir, f"{image_id}")
        # Try extensions
        loaded = False
        if not os.path.exists(image_path):
             for ext in ['.jpg', '.png', '.jpeg', '']: # Empty ext if already in ID
                 if os.path.exists(image_path + ext) and os.path.isfile(image_path + ext):
                     image_path += ext
                     loaded = True
                     break
        else:
            loaded = True
        
        try:
            if loaded:
                image = Image.open(image_path).convert('RGB')
                pixel_values = self.image_transform(image)
            else:
                # logger.warning(f"Image not found: {image_id}")
                pixel_values = torch.zeros((3, Config.IMAGE_SIZE, Config.IMAGE_SIZE))
        except Exception:
            pixel_values = torch.zeros((3, Config.IMAGE_SIZE, Config.IMAGE_SIZE))
            
        patches = self.patch_image(pixel_values) 
        
        # --- Sub-Module 1.4: Augmentation (Skip for strict dataset loading unless requested) ---
        if self.is_training and False: # Disabled for now to ensure base loading works first
             input_ids, patches = self.apply_augmentation(input_ids, patches)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'patches': patches,
            'label_ids': label_ids,
            'image_id': image_id
        }

    def patch_image(self, pixel_values):
        c, h, w = pixel_values.shape
        p = Config.PATCH_SIZE
        pixel_values = pixel_values.unsqueeze(0)
        patches = torch.nn.functional.unfold(pixel_values, kernel_size=(p, p), stride=(p, p)) 
        patches = patches.transpose(1, 2).squeeze(0) 
        return patches
    
    def apply_augmentation(self, input_ids, patches):
        # Placeholder for augmentation logic
        return input_ids, patches
