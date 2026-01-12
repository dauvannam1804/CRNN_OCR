
import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from dataset import OCRDataset, get_vocab
from model import CRNN
from utils import decode_greedy, calculate_accuracy

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

def get_args():
    parser = argparse.ArgumentParser(description='Train CRNN OCR')
    parser.add_argument('--data_root', type=str, default='data', help='path to data directory')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--hidden_size', type=int, default=256, help='hidden size of RNN')
    parser.add_argument('--imgW', type=int, default=100, help='image width')
    parser.add_argument('--imgH', type=int, default=32, help='image height')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='directory to save checkpoints')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    return parser.parse_args()

def train(model, device, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    
    for i, (images, labels, label_lengths) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.squeeze() # [B]
        
        batch_size = images.size(0)
        
        preds = model(images) # [T, B, C]
        
        preds_size = torch.LongTensor([preds.size(0)] * batch_size).to(device)
        
        loss = criterion(preds.log_softmax(2), labels, preds_size, label_lengths)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(train_loader)

def validate(model, device, val_loader, criterion, idx2char):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for i, (images, labels, label_lengths) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device) # flat [Sum(len)]
            label_lengths = label_lengths.squeeze() # [B]
            
            batch_size = images.size(0)
            
            preds = model(images) # [T, B, C]
            preds_size = torch.LongTensor([preds.size(0)] * batch_size).to(device)
            
            loss = criterion(preds.log_softmax(2), labels, preds_size, label_lengths)
            total_loss += loss.item()
            
            # Decode
            decoded_preds = decode_greedy(preds, idx2char)
            
            # Decode targets
            # labels is a flat tensor, need to split by label_lengths
            start = 0
            decoded_targets = []
            for length in label_lengths:
                length = length.item()
                label = labels[start : start + length]
                valid_char_indices = [idx.item() for idx in label]
                target_str = "".join([idx2char[idx] for idx in valid_char_indices])
                decoded_targets.append(target_str)
                start += length
            
            all_preds.extend(decoded_preds)
            all_targets.extend(decoded_targets)
            
    full_acc, char_acc = calculate_accuracy(all_preds, all_targets)
    return total_loss / len(val_loader), full_acc, char_acc

def main():
    args = get_args()
    set_seed(args.seed)
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(message)s',
                        handlers=[
                            logging.FileHandler(os.path.join(args.save_dir, "train.log")),
                            logging.StreamHandler()
                        ])
    logger = logging.getLogger()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    train_dir = os.path.join(args.data_root, 'trainset')
    test_dir = os.path.join(args.data_root, 'testset')
    
    # Save vocab
    with open(os.path.join(args.save_dir, "vocab.txt"), "w") as f:
        f.write("\n".join(vocab))
    logger.info(f"Saved vocab to {os.path.join(args.save_dir, 'vocab.txt')}")

    train_dataset = OCRDataset(train_dir, vocab, height=args.imgH, width=args.imgW)
    val_dataset = OCRDataset(test_dir, vocab, height=args.imgH, width=args.imgW)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.workers, collate_fn=OCRDataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.workers, collate_fn=OCRDataset.collate_fn)
    
    n_class = len(vocab) + 1 # +1 for blank
    model = CRNN(args.imgH, 1, n_class, args.hidden_size).to(device)
    
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_acc = 0.0
    
    logger.info("Starting training...")
    
    for epoch in range(args.epochs):
        train_loss = train(model, device, train_loader, criterion, optimizer)
        val_loss, full_acc, char_acc = validate(model, device, val_loader, criterion, train_dataset.idx2char)
        
        logger.info(f"Epoch [{epoch+1}/{args.epochs}] "
                    f"Train Loss: {train_loss:.4f} "
                    f"Val Loss: {val_loss:.4f} "
                    f"Full Acc: {full_acc:.4f} "
                    f"Char Acc: {char_acc:.4f}")
        
        if full_acc > best_acc:
            best_acc = full_acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pth"))
            logger.info("Saved best model.")
            
        # Also save latest
        torch.save(model.state_dict(), os.path.join(args.save_dir, "latest_model.pth"))

if __name__ == '__main__':
    main()
