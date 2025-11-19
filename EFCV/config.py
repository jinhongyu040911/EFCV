"""
EFCV Model Configuration

Configuration parameters for the EFCV model.
These settings correspond to the experimental setup described in the paper.
"""

class Config:
    # Model architecture
    d_model = 512          # Feature dimension (CLIP ViT-B/16)
    nhead = 8              # Number of attention heads
    num_layers = 3         # Number of Transformer layers
    dropout = 0.1          # Dropout rate
    
    # Training hyperparameters
    batch_size = 16        # Batch size (dataset-dependent)
    learning_rate = 4e-5   # Learning rate for AdamW
    weight_decay = 1e-4    # Weight decay for regularization
    
    # Training control
    max_epochs = 100       # Maximum training epochs
    patience = 10          # Early stopping patience
    
    # Dataset-specific sequence lengths
    # N: text entity sequence length (including global feature)
    # M: image entity sequence length (including global feature, fixed at 6)
    datasets = {
        'MR2_Chinese': {'text_seq_len': 26, 'img_seq_len': 6},
        'MR2_English': {'text_seq_len': 21, 'img_seq_len': 6},
        'Weibo': {'text_seq_len': 31, 'img_seq_len': 6},
        'PHEME': {'text_seq_len': 16, 'img_seq_len': 6}
    }
