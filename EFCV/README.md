# EFCV: Entity-aware Fusion of Consistency and Visual Clues

Official PyTorch implementation of "EFCV: Entity-aware Fusion of Consistency and Visual Clues for Multimodal Fake News Detection" (Neurocomputing 2025).

## Overview

EFCV is a multimodal fake news detection framework featuring:
- **Entity-aware dual consistency measurement**: Evaluates both intra-modal coherence and cross-modal alignment
- **Multi-scale feature fusion**: Integrates coarse-grained, fine-grained, and entity-level features through adaptive gating
- **Evidence theory-based fusion**: Combines dual-view evidence using Dirichlet distribution and uncertainty quantification

## Model Architecture

The EFCV framework consists of three main components:

1. **Consistency Module**: Computes entity-aware features through dual consistency measurement and multi-scale fusion
2. **Visual Clue Module**: Extracts visual evidence using weight-guided feature aggregation
3. **Fusion Module**: Adaptively combines evidence from both views using evidence theory

## Requirements

```bash
pip install torch torchvision numpy scikit-learn tqdm
```

Tested with:
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+

## Data Format

This implementation expects preprocessed entity features as input:

```python
{
    'text_features': torch.Tensor,  # Shape: [batch_size, N, 512]
    'img_features': torch.Tensor,   # Shape: [batch_size, M, 512]
    'label': torch.Tensor           # Shape: [batch_size]
}
```

Where:
- `N` is the text entity sequence length (dataset-specific: 16/21/26/31)
- `M` is the image entity sequence length (fixed at 6)
- Features are encoded using CLIP ViT-B/16

**Note**: Entity extraction and feature encoding are performed during data preprocessing. The model operates on pre-extracted entity features. For entity extraction methodology, please refer to Section 3.2 of our paper.

For detailed data format specifications, see [DATA_FORMAT.md](DATA_FORMAT.md).

## Training

```python
from models.efcv_model import create_efcv_model

model = create_efcv_model(
    d_model=512,
    nhead=8,
    num_layers=3,
    dropout=0.1
)
```

For complete training example, see `train.py`.

## Usage Example

```bash
# Train on preprocessed data
python train.py
```

Note: Modify the dataset path and hyperparameters in `train.py` as needed.

## Project Structure

```
EFCV/
├── models/
│   ├── efcv_model.py      # Core EFCV model implementation
│   └── __init__.py
├── utils/
│   ├── metrics.py         # Evaluation metrics
│   ├── early_stopping.py  # Early stopping mechanism
│   └── __init__.py
├── config.py              # Model configuration
├── train.py               # Training script
├── DATA_FORMAT.md         # Data format specification
└── README.md
```


## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{efcv2025,
  title={EFCV: Entity-aware Fusion of Consistency and Visual Clues for Multimodal Fake News Detection},
  author={Jin, Hongyu and Zhang, Mingshu and Bin, Wei and Zhang, Yuechuan and Wang, Yaxuan},
  journal={},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

