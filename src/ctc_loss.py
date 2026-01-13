
import torch
import torch.nn as nn

class CTCLossFromScratch(nn.Module):
    def __init__(self, blank=0, reduction='mean', zero_infinity=False):
        super(CTCLossFromScratch, self).__init__()
        self.blank = blank
        self.reduction = reduction
        self.zero_infinity = zero_infinity

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        """
        log_probs: [T, B, C] (Log Softmax of logits)
        targets: Flat targets [Sum(target_lengths)]
        input_lengths: [B]
        target_lengths: [B]
        """
        batch_size = log_probs.size(1)
        losses = []
        target_start = 0
        
        for b in range(batch_size):
            T = input_lengths[b].item()
            L = target_lengths[b].item()
            
            # 1. Prepare Data
            lp = log_probs[:T, b, :] # [T, C]
            current_target = targets[target_start : target_start + L]
            target_start += L
            
            # Extend Target
            extended_target = [self.blank]
            for char in current_target:
                extended_target.append(char.item())
                extended_target.append(self.blank)
            extended_target = torch.tensor(extended_target, dtype=torch.long, device=log_probs.device)
            S = len(extended_target)
            
            # 2. Dynamic Programming (Alpha Calculation)
            # Use list to accumulate steps for Autograd safety
            
            # Init t=0
            # Initialize with very small log probability (-inf)
            alpha_t = torch.full((S,), -float('inf'), device=log_probs.device)
            
            if T > 0:
                # Can start at blank (index 0) or first char (index 1)
                # We need to construct the first row carefully to maintain gradient connection
                # We can't just assign in-place like alpha_t[0] = ... if alpha_t is a leaf.
                # Constructing a list of values and stacking is safer but slower for S.
                # OR: masking. 
                
                # alpha_0[0] = lp[0, extended_target[0]]
                # alpha_0[1] = lp[0, extended_target[1]]
                # others = -inf
                
                # Let's use a mask/scatter approach or list-stack for the init row?
                # Actually, for small S (2*L+1 ~ 50-100), we can iterate? 
                # Better: create indices
                
                # Valid start indices: 0, 1
                start_indices = torch.tensor([0, 1] if S > 1 else [0], device=log_probs.device)
                start_probs = lp[0, extended_target[start_indices]] # [2]
                
                # Scatter these into alpha_t
                alpha_t = alpha_t.scatter(0, start_indices, start_probs)
                
            alpha_history = [alpha_t]
            
            for t in range(1, T):
                prev_alpha = alpha_history[-1]
                current_alpha_list = []
                
                # Vectorized operation for "next step" is hard because of dependencies (s-1, s-2)
                # But S dimension is usually small enough for iterative construction or shifted vector add?
                # Shifted approach:
                # Stay: prev_alpha[s] + lp[t, char_id]
                # Move: prev_alpha[s-1] + lp[t, char_id]
                # Skip: prev_alpha[s-2] + lp[t, char_id]
                
                # Let's look at shifts:
                # prev_stay = prev_alpha
                # prev_move = cat([-inf], prev_alpha[:-1])
                # prev_skip = cat([-inf, -inf], prev_alpha[:-2])
                
                # This seems efficient!
                
                # LogAddExp(a, b)
                
                # 1. Stay
                log_p = prev_alpha
                
                # 2. Move (Shift right by 1)
                prev_move = torch.cat([
                    torch.tensor([-float('inf')], device=log_probs.device), 
                    prev_alpha[:-1]
                ])
                log_p = torch.logaddexp(log_p, prev_move)
                
                # 3. Skip Blank (Shift right by 2)
                # Condition: extended_target[s] != blank AND extended_target[s] != extended_target[s-2]
                # We can compute a mask for this condition
                
                # Shifted Target comparison
                # s and s-2
                # Create mask: (target[s] != blank) & (target[s] != target[s-2])
                # Note: extended_target has blanks at even indices (0, 2, 4...)
                # So checks are only needed at odd indices (s>1)
                
                # Let's construct mask for skip
                # Only possible if S >= 3?
                
                if S > 2:
                    prev_skip = torch.cat([
                        torch.tensor([-float('inf'), -float('inf')], device=log_probs.device),
                        prev_alpha[:-2]
                    ])
                    
                    # Mask for skip validity
                    # We need a boolean tensor of shape [S]
                    # Logic: 
                    # 1. s >= 2 (Implicit in shift)
                    # 2. extended_target[s] != blank (Indices where extended_target != blank)
                    # 3. extended_target[s] != extended_target[s-2]
                    
                    chars = extended_target # [S]
                    chars_shifted = torch.cat([torch.tensor([-1, -1], device=log_probs.device), chars[:-2]])
                    
                    # Condition mask
                    # chars != 0 (blank is 0)
                    # chars != chars_shifted
                    skip_condition = (chars != self.blank) & (chars != chars_shifted)
                    
                    # Apply logaddexp only where condition is true
                    # where(condition, logaddexp(log_p, skip), log_p)
                    log_p = torch.where(skip_condition, torch.logaddexp(log_p, prev_skip), log_p)
                
                # Add emission probs
                # lp[t] shape [C]
                # We need probs for the extended_target characters
                # Gather emission probs for the current 's' characters
                emissions = lp[t].gather(0, extended_target) # [S]
                
                next_alpha = log_p + emissions
                alpha_history.append(next_alpha)
            
            # 3. Final Loss
            final_alpha = alpha_history[-1]
            if S > 1:
                # Sum of last two states (last blank or last char)
                # The valid end states are S-1 (final blank) and S-2 (final char)
                total_log_prob = torch.logaddexp(final_alpha[S-1], final_alpha[S-2])
            else:
                total_log_prob = final_alpha[S-1]
                
            nll = -total_log_prob
            if L > 0:
                nll = nll / L
            losses.append(nll)
            
        losses = torch.stack(losses)
        
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:
            return losses
