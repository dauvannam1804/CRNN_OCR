
import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class OCRDataset(Dataset):
    def __init__(self, root_dir, vocab, height=32, width=128, transform=None):
        self.root_dir = root_dir
        self.image_paths = glob.glob(os.path.join(root_dir, "*.jpeg")) + glob.glob(os.path.join(root_dir, "*.jpg"))
        self.vocab = vocab
        self.char2idx = {char: idx + 1 for idx, char in enumerate(vocab)} # 0 is reserved for blank
        self.idx2char = {idx + 1: char for idx, char in enumerate(vocab)}
        self.height = height
        self.width = width
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("L") # Convert to grayscale
        except IOError:
            print(f"Corrupted image: {img_path}")
            return self.__getitem__((idx + 1) % len(self))

        # Basic resizing generally used in CRNN
        image = image.resize((self.width, self.height), Image.Resampling.BILINEAR)

        if self.transform:
            image = self.transform(image)
        else:
            # Default transform if none provided
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            image = transform(image)

        label_str = os.path.splitext(os.path.basename(img_path))[0]
        
        # Filter characters not in vocab
        label_str = "".join([c for c in label_str if c in self.vocab])
        
        label = [self.char2idx[c] for c in label_str]
        label = torch.LongTensor(label)
        label_len = torch.LongTensor([len(label)])

        return image, label, label_len

    @staticmethod
    def collate_fn(batch):
        images, labels, label_lengths = zip(*batch)
        images = torch.stack(images, 0)
        labels = torch.cat(labels, 0)
        label_lengths = torch.cat(label_lengths, 0)
        return images, labels, label_lengths

def get_vocab(train_dir, test_dir):
    image_paths = glob.glob(os.path.join(train_dir, "*.jpeg")) + \
                  glob.glob(os.path.join(train_dir, "*.jpg")) + \
                  glob.glob(os.path.join(test_dir, "*.jpeg")) + \
                  glob.glob(os.path.join(test_dir, "*.jpg"))
    
    all_labels = [os.path.splitext(os.path.basename(p))[0] for p in image_paths]
    unique_chars = sorted(list(set("".join(all_labels))))
    return unique_chars

