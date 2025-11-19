"""
EFCV Training Script

Training script for the EFCV model on preprocessed entity features.

Requirements:
- Preprocessed entity features (see DATA_FORMAT.md)
- Entity extraction and CLIP encoding performed beforehand
- Dataset files in JSON format

Data format:
- text_features: [N, 512] - CLIP-encoded text entity features
- img_features: [M, 512] - CLIP-encoded image entity features  
- label: int - Binary label (0: fake, 1: real)

Usage:
    python train.py
    
Modify the 'main()' function to specify dataset and hyperparameters.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
import os
from pathlib import Path
from tqdm import tqdm

from models.efcv_model import create_efcv_model
from utils.metrics import calculate_metrics
from utils.early_stopping import EarlyStopping

class EFCVDataset(Dataset):
    def __init__(self, dataset_name, split):
        self.dataset_name = dataset_name
        self.split = split
        self.data = self._load_data()
        
    def _load_data(self):
        file_path = Path('datasets') / self.dataset_name / f'dataset_items_{self.split}.json'
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'text_features': torch.tensor(item['text_features'], dtype=torch.float32),
            'img_features': torch.tensor(item['img_features'], dtype=torch.float32),
            'label': torch.tensor(item['label'], dtype=torch.long)
        }

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Training"):
        batch_data = {
            'text_features': batch['text_features'].to(device),
            'img_features': batch['img_features'].to(device)
        }
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        output = model(batch_data)
        loss = nn.CrossEntropyLoss()(output['logits'], labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(output['logits'], dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    metrics = calculate_metrics(all_labels, all_preds)
    return total_loss / len(dataloader), metrics

def validate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            batch_data = {
                'text_features': batch['text_features'].to(device),
                'img_features': batch['img_features'].to(device)
            }
            labels = batch['label'].to(device)
            
            output = model(batch_data)
            preds = torch.argmax(output['logits'], dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    metrics = calculate_metrics(all_labels, all_preds)
    return metrics

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = create_efcv_model(
        d_model=512,
        nhead=8,
        num_layers=3,
        dropout=0.1,
        feature_out=32,
        num_classes=2
    ).to(device)
    
    train_dataset = EFCVDataset('MR2_Chinese', 'train')
    val_dataset = EFCVDataset('MR2_Chinese', 'val')
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    optimizer = optim.AdamW(model.parameters(), lr=4e-5, weight_decay=1e-4)
    early_stopping = EarlyStopping(patience=10)
    
    for epoch in range(100):
        train_loss, train_metrics = train_epoch(model, train_loader, optimizer, device)
        val_metrics = validate(model, val_loader, device)
        
        print(f"Epoch {epoch+1}")
        print(f"Train Loss: {train_loss:.4f}, Train F1: {train_metrics['f1']:.4f}")
        print(f"Val Acc: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['f1']:.4f}")
        
        if early_stopping(val_metrics['f1'], model):
            print("Early stopping triggered")
            break
    
    print("Training completed")

if __name__ == '__main__':
    main()
