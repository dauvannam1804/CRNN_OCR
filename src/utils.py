
import torch

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
