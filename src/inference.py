
import os
import argparse
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from model import CRNN
from dataset import get_vocab
from utils import decode_greedy

def get_args():
    parser = argparse.ArgumentParser(description='Inference CRNN OCR')
    parser.add_argument('--image_path', type=str, required=True, help='path to image or directory')
    parser.add_argument('--model_path', type=str, required=True, help='path to trained model')
    parser.add_argument('--data_root', type=str, default='data', help='path to data directory to get vocab')
    parser.add_argument('--imgW', type=int, default=100, help='image width')
    parser.add_argument('--imgH', type=int, default=32, help='image height')
    parser.add_argument('--hidden_size', type=int, default=256, help='hidden size of RNN')
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    # Try loading vocab from the same directory as the model
    model_dir = os.path.dirname(args.model_path)
    vocab_path = os.path.join(model_dir, "vocab.txt")
    
    if os.path.exists(vocab_path):
        print(f"Loading vocab from {vocab_path}")
        with open(vocab_path, "r") as f:
            vocab = f.read().splitlines()
    else:
        print(f"vocab.txt not found in {model_dir}, scanning data directory...")
        train_dir = os.path.join(args.data_root, 'trainset')
        test_dir = os.path.join(args.data_root, 'testset')
        vocab = get_vocab(train_dir, test_dir)
    
    if len(vocab) == 0:
        raise ValueError("Vocabulary is empty! Make sure 'vocab.txt' exists or data directories are correct.")
        
    print(f"Vocab size: {len(vocab)}")
    idx2char = {idx + 1: char for idx, char in enumerate(vocab)}
    n_class = len(vocab) + 1

    model = CRNN(args.imgH, 1, n_class, args.hidden_size).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    if os.path.isdir(args.image_path):
        import glob
        image_paths = glob.glob(os.path.join(args.image_path, "*.jpeg")) + glob.glob(os.path.join(args.image_path, "*.jpg"))
    else:
        image_paths = [args.image_path]
        
    for path in image_paths:
        image = Image.open(path).convert('L')
        # Resize to fixed height, fixed width (same as training)
        image = image.resize((args.imgW, args.imgH), Image.Resampling.BILINEAR)
        image = transform(image)
        image = image.unsqueeze(0).to(device) # [1, 1, H, W]
        
        with torch.no_grad():
            preds = model(image) # [T, 1, C]
            decoded_text = decode_greedy(preds, idx2char)[0]
            
        print(f"{path}: {decoded_text}")

if __name__ == '__main__':
    main()
