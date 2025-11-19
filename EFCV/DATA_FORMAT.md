# Data Format Specification

## Overview

The EFCV model operates on **preprocessed entity features** extracted from raw multimodal data. This document describes the expected input format.

## Input Format

### Training Data Structure

Each sample should contain:

```python
{
    'text_features': torch.Tensor,  # Shape: [N, 512]
    'img_features': torch.Tensor,   # Shape: [M, 512]
    'label': int                     # 0 (fake) or 1 (real)
}
```

### Feature Dimensions

| Component | Dimension | Description |
|-----------|-----------|-------------|
| `text_features` | [N, 512] | Text entity features (CLIP ViT-B/16 encoded) |
| `img_features` | [M, 512] | Image entity features (CLIP ViT-B/16 encoded) |
| `label` | scalar | Binary label |

Where:
- **N**: Text entity sequence length (dataset-dependent)
  - PHEME: 16
  - MR2_English: 21
  - MR2_Chinese: 26
  - Weibo: 31
- **M**: Image entity sequence length (fixed at 6)

## Feature Encoding

All features should be:
1. **Extracted** using appropriate entity extraction methods
2. **Encoded** using CLIP ViT-B/16 (output dimension: 512)
3. **Normalized** using L2 normalization (applied within the model)

### Sequence Structure

Both text and image sequences follow the same structure:

```
Position 0: Global feature (entire text/image)
Position 1-N/M-1: Entity features (individual entities)
```

For example, with M=6 image entities:
```
img_features[0]: Global image feature
img_features[1:6]: 5 individual entity features
```

## Preprocessing Pipeline

The preprocessing pipeline (not included in this repository) should:

1. **Entity Extraction**:
   - Text: Extract named entities, nouns, proper nouns, adjectives, and verbs
   - Image: Detect objects using object detection methods

2. **Entity Selection**:
   - Apply entity quantity limits
   - Rank entities by importance
   - Pad or truncate to fixed length

3. **Feature Encoding**:
   - Encode entities using CLIP
   - Include global feature as first element
   - Save as PyTorch tensors

For detailed methodology, refer to **Section 3.2** of the paper.

## Dataset Files

Expected file structure:
```
datasets/
├── MR2_Chinese/
│   ├── dataset_items_train.json
│   ├── dataset_items_val.json
│   └── dataset_items_test.json
├── MR2_English/
│   └── ...
├── Weibo/
│   └── ...
└── PHEME/
    └── ...
```

Each JSON file contains a list of preprocessed samples.

## Notes

- All features are expected to be **pre-extracted** before training
- The model does **not** perform entity extraction or CLIP encoding
- Feature extraction should follow the methodology described in the paper (Section 3.2)
- Entity extraction uses Faster R-CNN (images) and SpaCy (text)
- Feature encoding uses CLIP ViT-B/16 model
