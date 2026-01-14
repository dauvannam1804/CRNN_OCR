
import torch
import matplotlib.pyplot as plt
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math
from collections import defaultdict

def decode_greedy(logits, idx2char):
    """
    logits: [T, B, C] (Sequence length, Batch size, Num classes)
    idx2char: dictionary mapping index to character
    """
    # Get max index
    preds = logits.argmax(2) # [T, B]
    preds = preds.transpose(1, 0).contiguous() # [B, T]
    
    decoded_batch = []
    for pred in preds:
        decoded_str = []
        last_char = 0
        for char_idx in pred:
            char_idx = char_idx.item()
            if char_idx != last_char:
                if char_idx != 0: # 0 is blank
                    decoded_str.append(idx2char[char_idx])
            last_char = char_idx
        decoded_batch.append("".join(decoded_str))
    
    return decoded_batch

def decode_beam_search(logits, idx2char, beam_width=10):
    """
    logits: [T, B, C] (Sequence length, Batch size, Num classes)
    idx2char: dictionary mapping index to character
    beam_width: int, number of beams to keep
    """
    T, B, C = logits.shape
    # Softmax to get probabilities
    probs = torch.softmax(logits, dim=2)
    probs = probs.cpu().detach().numpy() # [T, B, C]
    
    decoded_batch = []
    
    for b in range(B):
        # Initialize beam: {(sequence_tuple): (prob_blank, prob_non_blank)}
        # Empty sequence has prob_blank=1.0, prob_non_blank=0.0
        beam = {(): (1.0, 0.0)}
        
        for t in range(T):
            next_beam = defaultdict(lambda: (0.0, 0.0))
            p_t = probs[t, b] # [C]
            
            for seq, (p_b, p_nb) in beam.items():
                # 1. Blank
                # Extending with blank: current end stays same.
                # If we ended in blank, we multiply by p_blank.
                # If we ended in non-blank, we multiply by p_blank (and now effective end is blank).
                # New score for this seq (ending in blank) is increased.
                p_blank = p_t[0]
                n_p_b, n_p_nb = next_beam[seq]
                n_p_b += (p_b + p_nb) * p_blank
                next_beam[seq] = (n_p_b, n_p_nb)
                
                # 2. Non-Blank
                # We can iterate over all characters or just top k to save time if C is large.
                # Here C is small (~39), so we iterate all.
                for c_idx in range(1, C):
                    p_char = p_t[c_idx]
                    char = idx2char[c_idx]
                    
                    if len(seq) > 0 and seq[-1] == char:
                        # Case A: Same character.
                        # 1. Transitions from non-blank (same char) -> Merge. Update non-blank score.
                        n_p_b_merge, n_p_nb_merge = next_beam[seq]
                        n_p_nb_merge += p_nb * p_char
                        next_beam[seq] = (n_p_b_merge, n_p_nb_merge)
                        
                        # 2. Transitions from blank -> New char (duplicate chars in text).
                        new_seq = seq + (char,)
                        n_p_b_new, n_p_nb_new = next_beam[new_seq]
                        n_p_nb_new += p_b * p_char
                        next_beam[new_seq] = (n_p_b_new, n_p_nb_new)
                    else:
                        # Case B: Different character or empty seq.
                        # Extend sequence.
                        new_seq = seq + (char,)
                        n_p_b_new, n_p_nb_new = next_beam[new_seq]
                        n_p_nb_new += (p_b + p_nb) * p_char
                        next_beam[new_seq] = (n_p_b_new, n_p_nb_new)
            
            # Prune beam
            # Score = p_b + p_nb
            sorted_beam = sorted(next_beam.items(), key=lambda x: x[1][0] + x[1][1], reverse=True)
            beam = dict(sorted_beam[:beam_width])
            
        # Get best path
        best_seq, _ = max(beam.items(), key=lambda x: x[1][0] + x[1][1])
        decoded_batch.append("".join(best_seq))
        
    return decoded_batch

def calculate_accuracy(predictions, targets):
    """
    predictions: list of strings
    targets: list of strings
    """
    n_samples = len(predictions)
    full_match = 0
    total_char_dist = 0
    total_chars = 0
    
    for pred, target in zip(predictions, targets):
        if pred == target:
            full_match += 1
            
        # Character match (simple positional match for same length, or Levenshtein could be better but requester asked for "match cung theo tung vi tri")
        # "match cứng theo từng vị trí kí tự" -> Hard match by position.
        # This implies we compare char by char. If lengths differ, we can limit to min length.
        
        l_pred = len(pred)
        l_target = len(target)
        min_len = min(l_pred, l_target)
        
        match_chars = 0
        for i in range(min_len):
            if pred[i] == target[i]:
                match_chars += 1
        
        total_char_dist += match_chars
        total_chars += max(l_pred, l_target) # Or should it be l_target? Usually accuracy is matches / ground_truth_len. 
        # But "match cứng theo từng vị trí" might imply we should penalize extra/missing chars.
        # Let's use total chars in target as denominator for "char accuracy" or similar.
        # However, a common metric is Edit Distance. But the user asked for "Match char (match cứng theo từng vị trí kí tự)".
        # Let's assume matches / max(len(pred), len(target)) for per-sample char accuracy, averaged?
        # Or Sum(matches) / Sum(max(lengths)).
        
    full_acc = full_match / n_samples
    
    # Let's define Char Match Accuracy as: Sum(matches) / Sum(max(len(pred), len(target))) to strictly penalize length mismatch too?
    # Or Sum(matches) / Sum(len(target))? 
    # Let's stick to Sum(matches) / Sum(len(target)) generally, but purely positional match.
    
    target_chars = sum([len(t) for t in targets])
    if target_chars == 0:
        char_acc = 0
    else:
        # Re-calc matches purely based on target length coverage
        matches = 0
        for pred, target in zip(predictions, targets):
            for i in range(min(len(pred), len(target))):
                if pred[i] == target[i]:
                    matches += 1
        char_acc = matches / target_chars

    return full_acc, char_acc

def plot_training_history(csv_path, save_path='training_history.png'):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Expected columns: epoch, train_loss, val_loss, full_acc, char_acc
    
    plt.figure(figsize=(18, 5))
    
    # 1. Train & Val Loss
    plt.subplot(1, 3, 1)
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.legend()
    
    # 2. Full Match Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(df['epoch'], df['full_acc'], label='Full Match Acc', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Full Sequence Accuracy')
    plt.legend()
    
    # 3. Char Accuracy
    plt.subplot(1, 3, 3)
    plt.plot(df['epoch'], df['char_acc'], label='Char Acc', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Character Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training history plot saved to {save_path}")

