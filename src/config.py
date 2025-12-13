import torch

class Config:
    # Paths
    DATA_DIR = "./data/NER_data"
    
    # Twitter-15 Labels
    LABELS = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-OTHER", "I-OTHER"]
    LABEL2ID = {label: i for i, label in enumerate(LABELS)}
    ID2LABEL = {i: label for i, label in enumerate(LABELS)}
    NUM_LABELS = len(LABELS)
    
    # Hyperparameters (from Spec)
    MAX_SEQ_LEN = 128      # N
    HIDDEN_DIM = 768       # D
    PATCH_SIZE = 16        # P_S
    IMAGE_SIZE = 224
    
    # Calculated
    NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # P = 196
    PATCH_DIM = PATCH_SIZE * PATCH_SIZE * 3        # D_patch = 768
    
    # Model Config
    DROPOUT = 0.1
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Tokenizer
    BERT_MODEL_NAME = "bert-base-uncased"
