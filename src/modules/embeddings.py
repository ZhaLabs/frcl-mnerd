import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from src.config import Config

class TextEmbedding(nn.Module):
    """
    Sub-Module 1.3 Steps 1 & 2: Text Projection & Positional Encoding
    
    In this implementation, we leverage the pre-trained BERT embeddings 
    which strictly contain both the Lookup Table and Learned Positional Embeddings.
    """
    def __init__(self):
        super().__init__()
        # We load the full BERT model but primarily use its embedding layer
        self.bert = BertModel.from_pretrained(Config.BERT_MODEL_NAME)
        
        # If we stricly want ONLY the embeddings as per spec:
        # T_proj = Lookup(TokenIDs)
        # T_Emb = T_proj + Positional
        # self.bert.embeddings does exactly this in PyTorch
        self.embeddings = self.bert.embeddings

    def forward(self, input_ids):
        """
        Args:
            input_ids: [Batch, N]
        Returns:
            T_Emb: [Batch, N, D]
        """
        # Step 1 & 2 combined (Lookup + Position + Segment/TokenParams)
        return self.embeddings(input_ids)


class VisualEmbedding(nn.Module):
    """
    Sub-Module 1.3 Steps 3 & 4: Visual Projection & Positional Encoding
    """
    def __init__(self):
        super().__init__()
        self.patch_dim = Config.PATCH_DIM  # 768
        self.hidden_dim = Config.HIDDEN_DIM # 768
        self.num_patches = Config.NUM_PATCHES # 196
        
        # Step 3: Visual Projection (Linear Layer)
        # Projects [D_patch] -> [D]
        self.projection = nn.Linear(self.patch_dim, self.hidden_dim)
        
        # Step 4: Visual Positional Encoding (Learned)
        # Parameter shape: [1, P, D] to broadcast over batch
        self.positional_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches, self.hidden_dim)
        )
        
        # Initialize weights (trunc_normal is standard for ViT)
        nn.init.trunc_normal_(self.positional_embedding, std=0.02)
        
    def forward(self, patches):
        """
        Args:
            patches: Raw Patch Vectors [Batch, P, D_patch]
        Returns:
            V_Emb: [Batch, P, D]
        """
        # Step 3: Projection
        v_proj = self.projection(patches) # [Batch, P, D]
        
        # Step 4: Add Positional Embeddings
        # Element-wise addition
        v_emb = v_proj + self.positional_embedding
        
        return v_emb


class MultimodalPreprocessorModule(nn.Module):
    """
    Wrapper Module that encapsulates the entire Sub-Module 1.3 
    to return the FINAL OUTPUTS specified.
    """
    def __init__(self):
        super().__init__()
        self.text_embedder = TextEmbedding()
        self.visual_embedder = VisualEmbedding()
        
    def forward(self, input_ids, patches):
        """
        Takes the outputs from the Dataset (Sub-module 1.1 & 1.2)
        and produces the Final Embeddings (Sub-module 1.3).
        """
        # 1. T_Emb
        t_emb = self.text_embedder(input_ids)
        
        # 2. V_Emb
        v_emb = self.visual_embedder(patches)
        
        return t_emb, v_emb
