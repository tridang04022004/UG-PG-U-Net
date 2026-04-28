# Uncertainty Guided Progressive Growing U-Net

A deep learning model for nuclei segmentation combining progressive growing training with uncertainty-guided iterative refinement. Built from the progressive growing backbone of PGU-Net+ with enhanced uncertainty guidance capabilities.

## References

This project is built upon the progressive growing backbone of:

- **PGU-Net+**: [Progressive Growing U-Net for Segmentation of the Prostate in Transrectal Ultrasound Images](https://arxiv.org/abs/1911.01062)

## Key Features

- **Progressive Growing**: Gradually increases model complexity and input resolution (32×32 → 64×64 → 128×128 → 256×256)
- **Uncertainty Guidance**: Uses Shannon entropy to identify uncertain regions and guide iterative refinement
- **Multi-loss Training**: Combines focal loss, dice loss, and boundary loss for robust segmentation
- **Test-Time Augmentation (TTA)**: Optional TTA for improved robustness during inference
- **IoU Plateau Detection**: Adaptive boundary refinement when IoU improvement stalls

## Dataset

Download the Herlev Nuclei dataset from:
https://mde-lab.aegean.gr/index.php/downloads/

The dataset contains:

- Binary masks with class labels:
  - Class 0: Background + Unknown regions
  - Class 1: Small nuclei (RGB: [0, 0, 255])
  - Class 2: Large nuclei (RGB: [0, 0, 128])

## Setup and Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd UGPGUnetPlus
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Prepare Dataset

1. Download the dataset from the link above
2. Extract and organize the data:

```
data/
├── train/
│   ├── image1.BMP
│   ├── image1-d.bmp
│   ├── image2.BMP
│   ├── image2-d.bmp
│   └── ...
└── test/
    ├── image_test1.BMP
    ├── image_test1-d.bmp
    ├── image_test2.BMP
    ├── image_test2-d.bmp
    └── ...
```

## Running the Model

### Basic Training

```bash
python train_PG.py \
    --data-dir ./data \
    --epochs 200 \
    --batch-size 16 \
    --lr 3e-4 \
    --max-stage 4
```

### With Uncertainty Guidance

```bash
python train_PG.py \
    --data-dir ./data \
    --epochs 200 \
    --batch-size 16 \
    --lr 3e-4 \
    --max-stage 4 \
    --use-uncertainty-guidance
```

### With Test-Time Augmentation

```bash
python train_PG.py \
    --data-dir ./data \
    --epochs 200 \
    --batch-size 16 \
    --lr 3e-4 \
    --max-stage 4 \
    --use-tta \
    --tta-mode standard
```

### Full Configuration Example

```bash
python train_PG.py \
    --data-dir ./data \
    --epochs 200 \
    --stage-epochs 40 \
    --max-stage 4 \
    --batch-size 16 \
    --lr 3e-4 \
    --disable-plateau-detection \
    --plateau-patience 7 \
    --plateau-min-delta 1e-4 \
    --plateau-boundary-boost 0.15 \
    --plateau-lr-factor 0.5 \
    --num-workers 4 \
    --output-dir ./outputs_PG \
    --use-uncertainty-guidance \
    --use-tta \
    --tta-mode standard
```

## Command Line Arguments

| Argument                      | Type  | Default        | Description                                             |
| ----------------------------- | ----- | -------------- | ------------------------------------------------------- |
| `--data-dir`                  | str   | `./data`       | Path to dataset directory                               |
| `--epochs`                    | int   | 200            | Total training epochs                                   |
| `--stage-epochs`              | int   | 40             | Epochs per progressive stage                            |
| `--max-stage`                 | int   | 4              | Maximum progressive stage (1-4)                         |
| `--batch-size`                | int   | 16             | Training batch size                                     |
| `--lr`                        | float | 3e-4           | Learning rate                                           |
| `--disable-plateau-detection` | flag  | -              | Disable IoU plateau detection                           |
| `--plateau-patience`          | int   | 7              | Epochs to wait before plateau trigger                   |
| `--plateau-min-delta`         | float | 1e-4           | Minimum improvement threshold                           |
| `--plateau-boundary-boost`    | float | 0.15           | Boundary loss increase at plateau                       |
| `--plateau-lr-factor`         | float | 0.5            | Learning rate multiplier at plateau                     |
| `--num-workers`               | int   | 0              | DataLoader workers                                      |
| `--output-dir`                | str   | `./outputs_PG` | Output directory for checkpoints                        |
| `--checkpoint`                | str   | None           | Path to resume training from                            |
| `--use-uncertainty-guidance`  | flag  | -              | Enable uncertainty-guided refinement                    |
| `--use-tta`                   | flag  | -              | Enable test-time augmentation                           |
| `--tta-mode`                  | str   | `standard`     | TTA mode: standard, flips_only, rotations_only, minimal |

## Uncertainty Guidance

The model uses Shannon entropy to compute uncertainty maps for iterative refinement:

```
Entropy = -sum(p * log(p))
Uncertainty = Entropy / log(num_classes)
```

Uncertain regions are amplified in subsequent iterations to refine predictions in challenging areas.

## Model Architecture

### Progressive Stages

1. **Stage 1** (32×32): UNet1 - Light-weight encoder-decoder
2. **Stage 2** (64×64): UNet2 - Intermediate model with residual modules
3. **Stage 3** (128×128): UNet3 - Deeper architecture
4. **Stage 4** (256×256): UNet4 - Full resolution model

All stages use residual modules instead of skip connections for independent feature learning.

### Loss Function

Combined loss with three components:

```
Loss = 0.475 * FocalLoss + 0.475 * DiceLoss + 0.05 * BoundaryLoss
```

- **Focal Loss**: Handles class imbalance
- **Dice Loss**: Optimizes intersection over union
- **Boundary Loss**: Emphasizes correct object boundaries

## Output Structure

Training creates the following outputs in `output_dir`:

```
outputs_PG/
├── latest_checkpoint_PG.pt         # Latest model state
├── best_model_PG.pt                # Best model (highest IoU)
├── stage1_checkpoint.pt
├── stage2_checkpoint.pt
├── stage3_checkpoint.pt
└── stage4_checkpoint.pt
```

## Training Monitoring

Training metrics printed at each epoch:

- **Loss**: Combined loss value
- **F1 Score**: Harmonic mean of precision and recall
- **Dice**: Sørensen-Dice coefficient
- **IoU**: Intersection over Union
- **Uncertainty**: Mean entropy across predictions

## Troubleshooting

### Out of Memory Error

- Reduce `--batch-size`
- Reduce `--num-workers`
- Use fewer TTA transforms (`--tta-mode minimal`)

### Poor Convergence

- Increase `--stage-epochs` for longer training per stage
- Reduce `--lr` (e.g., 1e-4 for finer tuning)
- Disable plateau detection: `--disable-plateau-detection`

### Dataset Not Found

Ensure dataset directory structure matches:

```
data/
├── train/       # Training images and masks
└── test/        # Test images and masks
```

Image files must be `.BMP` format with mask files suffixed with `-d.bmp`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For issues, questions, or contributions, please open an issue on the repository.
