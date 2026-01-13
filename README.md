# CRNN OCR Project

This project implements a CRNN (Convolutional Recurrent Neural Network) for OCR tasks from scratch using PyTorch.

## 1. Setup

This project uses `uv` for efficient environment management.

```bash
# Create virtual environment
uv venv .venv

# Activate environment
source .venv/bin/activate

# Install dependencies
uv pip install torch torchvision numpy Pillow matplotlib pandas
```

## 2. Data Preparation

1.  **Download Data**: Download the dataset from Kaggle:
    [Captcha Image Data](https://www.kaggle.com/datasets/sandeep1507/captchaimgdata/data)

2.  **Extract & Organize**: Extract the contents and organize them into a `data` folder at the project root. The structure should be:

    ```
    data/
    ├── trainset/       # Place training images here
    └── testset/        # Place test images here
    ```

    *Note: The system assumes the filename (without extension) is the ground truth label (e.g., `2B2847.jpeg` -> label `2B2847`).*

## 3. Training

To train the model, run the `src/train.py` script.

```bash
python3 src/train.py \
    --data_root ./data \
    --batch_size 64 \
    --epochs 50 \
    --lr 0.001 \
    --save_dir ./checkpoints
```

*   **Logs**: Training logs are printed to console and saved to `checkpoints/train.log`.
*   **Checkpoints**: The best model (based on validation accuracy) is saved as `best_model.pth`.
*   **Plots**: Training history plots (loss and accuracy) are saved as `checkpoints/training_history.png`.

### 3b. Training with Custom CTC Loss

To train using the custom Python implementation of CTC Loss (for educational purposes or debugging):

```bash
python3 src/train_custom.py \
    --data_root ./data \
    --batch_size 64 \
    --epochs 50 \
    --lr 0.001 \
    --save_dir ./checkpoints
```

*   **Logs**: Saved to `checkpoints/training_log_custom.csv`.
*   This script uses `src/ctc_loss.py` which implements the Forward Algorithm from scratch using PyTorch operations.

## 4. Inference

You can run inference on a single image or a whole directory.

**Single Image:**
```bash
python3 src/inference.py \
    --image_path data/testset/2AK279.jpeg \
    --model_path checkpoints/best_model.pth
```

**Directory:**
```bash
python3 src/inference.py \
    --image_path data/testset \
    --model_path checkpoints/best_model.pth
```