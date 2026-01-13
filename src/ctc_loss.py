
import torch
import torch.nn as nn
import torch.autograd as autograd

def _compute_alpha_matrix(log_probs, extended_targets, input_lengths, target_lengths, blank=0):
    """
    Computes Alpha Matrix (Forward)
    log_probs: [T, B, C]
    extended_targets: [B, S_max]
    """
    T_max, B, C = log_probs.shape
    S_max = extended_targets.size(1)
    device = log_probs.device
    
    alpha = torch.full((B, S_max), -float('inf'), device=device)
    
    # Init t=0
    # alpha[b, 0] = log_probs[0, b, blank]
    # alpha[b, 1] = log_probs[0, b, target[0]] (if valid)
    alpha[:, 0] = log_probs[0, :, blank]
    
    has_char = (target_lengths > 0)
    if has_char.any():
        # Scatter alpha[:, 1]
        first_char_indices = extended_targets[:, 1].unsqueeze(1) # [B, 1]
        first_char_probs = log_probs[0].gather(1, first_char_indices).squeeze(1)
        # Only update where has_char is true
        alpha[:, 1] = torch.where(has_char, first_char_probs, alpha[:, 1])

    alpha_history = torch.full((T_max, B, S_max), -float('inf'), device=device)
    alpha_history[0] = alpha

    # Loop
    # Pre-compute masks for optimization? 
    # Skip mask depends on s. 
    # skip_mask: [B, S_max]
    # (targets[s] != blank) & (targets[s] != targets[s-2])
    # This is constant for all t.
    targets_shifted_2 = torch.cat([torch.full((B, 2), -1, dtype=torch.long, device=device), extended_targets[:, :-2]], dim=1)
    skip_permission_mask = (extended_targets != blank) & (extended_targets != targets_shifted_2)

    for t in range(1, T_max):
        prev_alpha = alpha_history[t-1]
        lp_t = log_probs[t]
        emissions = lp_t.gather(1, extended_targets)
        
        # Stay
        log_p = prev_alpha
        
        # Move (Shift Right 1)
        prev_move = torch.cat([torch.full((B, 1), -float('inf'), device=device), prev_alpha[:, :-1]], dim=1)
        log_p = torch.logaddexp(log_p, prev_move)
        
        # Skip (Shift Right 2)
        prev_skip = torch.cat([torch.full((B, 2), -float('inf'), device=device), prev_alpha[:, :-2]], dim=1)
        log_p = torch.where(skip_permission_mask, torch.logaddexp(log_p, prev_skip), log_p)
        
        alpha_history[t] = log_p + emissions

    return alpha_history

def _compute_beta_matrix(log_probs, extended_targets, input_lengths, target_lengths, blank=0):
    """
    Computes Beta Matrix (Backward)
    log_probs: [T, B, C]
    extended_targets: [B, S_max]
    """
    T_max, B, C = log_probs.shape
    S_max = extended_targets.size(1)
    device = log_probs.device
    
    beta_history = torch.full((T_max, B, S_max), -float('inf'), device=device)
    
    # Init at T-1 (per sample)
    # But we run vectorized from T_max-1.
    # To handle variable T, we can run backwards from T_max-1.
    # Logic: beta needs to be initialized at input_lengths[b]-1.
    # For steps > input_lengths[b]-1, beta doesn't matter (masked out later).
    # But to utilize vectorization, we should treat padded steps as "Stay" transitions with 0 cost?
    # Or just run normally (log_probs might be garbage/padding) and fix init?
    # Better: Initialize beta at T_max-1.
    # But for a sample with len T < T_max, its valid beta starts at T.
    # Let's initialize a "running beta" of -inf.
    # At t = input_lengths[b]-1, we force inject the start condition.
    
    # Actually, simpler: Compute for all, mask results.
    # But backward recusrion depends on t+1.
    # We can mask "injection" of beta_init at t == input_lengths[b] - 1.
    
    beta = torch.full((B, S_max), -float('inf'), device=device)
    
    # Constant Skip Mask
    targets_shifted_2 = torch.cat([torch.full((B, 2), -1, dtype=torch.long, device=device), extended_targets[:, :-2]], dim=1)
    # Note: Beta skip is s -> s+2. 
    # Conditions: target[s+2] (which is next) != blank and target[s+2] != target[s]
    # But usually expressed as: transition s -> s' valid?
    # From s to s: always
    # From s to s-1: always
    # From s to s-2: if ...
    # Here we go backwards: from s' to s.
    # s' is t+1 state. s is t state.
    # Transitions coming OUT of s:
    # s -> s (Stay)
    # s -> s+1 (Next)
    # s -> s+2 (Skip)
    # We accumulate FROM targets:
    # beta[s] += beta[s] (Stay)
    # beta[s] += beta[s+1] (Next)
    # beta[s] += beta[s+2] (Skip)
    
    # Skip condition for s -> s+2:
    # target[s+2] != blank & target[s+2] != target[s]
    # Let's define mask at 's'.
    # Mask[s] is true if s -> s+2 is valid.
    # Need target[s+2] ...
    # extended_targets[:, 2:] compared to extended_targets[:, :-2]
    # We can precompute this.
    
    # Shift targets Left by 2
    targets_left_2 = torch.cat([extended_targets[:, 2:], torch.full((B, 2), -1, dtype=torch.long, device=device)], dim=1)
    # target[s+2] != blank (which is target_left_2)
    # target[s+2] != target[s]
    skip_permission_mask = (targets_left_2 != blank) & (targets_left_2 != extended_targets)
    
    # Loop t from T_max-1 down to 0
    for t in range(T_max - 1, -1, -1):
        # 1. Check if this t corresponds to T-1 for any batch
        # If t == input_lengths[b] - 1, we reset beta for that batch to Init state.
        
        # Init State: beta[S-1] = 0, beta[S-2] = 0 (if s-2 valid), others -inf.
        # S depends on batch.
        
        # Identify batches starting now
        is_last_step = (input_lengths == (t + 1))
        
        if is_last_step.any():
            # Create init beta for these batches
            # We construct a specific init tensor [B, S_max]
            # S_final = 2*L + 1. Indices: 0 to 2*L.
            # Valid ends: 2*L (blank) and 2*L-1 (char).
            # Note: 2*L is S-1 in 0-indexed terms?
            # My S_max was 2*L_max + 1. S_b = 2*L_b + 1.
            # End indices: S_b - 1 and S_b - 2.
            
            # Scatter 0.0 (log 1) to these indices
            
            # We can modify 'beta' in place for these batches
            # Reset to -inf first? No, previous iterations (t > T) were garbage/padding.
            # Just overwrite.
            
            # Mask for indices
            # Create a coordinate list?
            # Or just iterative update for simplicity since B is small?
            # Masked assignment is cleaner.
            
            # Default init: -inf
            init_beta = torch.full((B, S_max), -float('inf'), device=device)
            
            # 1. Last Blank: index 2*L
            last_blank_idx = 2 * target_lengths
            # 2. Last Char: index 2*L - 1
            last_char_idx = 2 * target_lengths - 1
            
            # Assign
            # index specific batch rows
            # We can use scatter_.
            
            # scatter src must be same size or scalar broadcast? PyTorch scatter supports scalar.
            init_beta.scatter_(1, last_blank_idx.unsqueeze(1), 0.0)
            
            # Only valid if L > 0 for last char
            has_char = (target_lengths > 0)
            if has_char.any():
                # We need to only scatter where has_char
                # Create temp scalar 0.0?
                # scatter 0.0 to all, but mask back?
                # safer:
                val = torch.zeros((B, 1), device=device)
                idx = last_char_idx.unsqueeze(1)
                # Scatter 0 to all indices (even invalid ones where idx=-1? check bounds)
                # last_char_idx can be -1 if L=0. Scatter handles indices.
                # Avoid negative indices if L=0.
                idx = idx.clone()
                idx[idx < 0] = 0 # Safe dummy
                
                # Create mask to apply
                # Actually, simpler:
                # init_beta.scatter(1, idx, 0.0) -> sets dummy too
                # Reset dummy? 
                # Or just loop over the "turning on" batches?
                pass 
                
            # Let's do a masked blend
            # For batches where is_last_step is True, replace beta with init_beta
            # But we generated init_beta for ALL batches.
            # We only want to use it for 'is_last_step' batches.
            
            # Re-generate init_beta properly for all
            init_beta = torch.full((B, S_max), -float('inf'), device=device)
            init_beta.scatter_(1, last_blank_idx.unsqueeze(1), 0.0)
            
            safe_char_idx = last_char_idx.unsqueeze(1).clone()
            safe_char_idx[safe_char_idx < 0] = 0
            
            # Temp tensor to scatter
            temp_char = torch.full((B, 1), -float('inf'), device=device)
            temp_char[has_char] = 0.0
            init_beta.scatter_(1, safe_char_idx, temp_char)
            
            # Mix: if is_last_step, use init_beta, else use computed beta
            beta = torch.where(is_last_step.unsqueeze(1), init_beta, beta)

        # Save this beta as "beta at t" (actually, this is beta "entering" t)
        # Wait, beta_t depends on beta_{t+1}.
        # The loop flows:
        # Start with beta at T (initially garbage/init).
        # Determine beta at T-1 from T.
        # But for the exact step T-1, we Overwrite with Init.
        # Then calculate Beta at T-2 from Beta T-1.
        
        # So:
        # 1. Update/Inject Init conditions (at current t).
        beta_history[t] = beta
        
        if t > 0: # Compute beta for t-1
            # Weighted Beta from t (next step relative to t-1)
            # Add emissions at t
            lp_t = log_probs[t]
            emissions = lp_t.gather(1, extended_targets)
            
            weighted_beta = beta + emissions
            
            # Transitions to t-1
            # Stay: s -> s
            log_p = weighted_beta
            
            # Next: s -> s+1 (s+1 in t maps to s in t-1)
            # Shift Left 1
            # [0, 1, 2, .] -> [1, 2, ., .]
            prev_next = torch.cat([weighted_beta[:, 1:], torch.full((B, 1), -float('inf'), device=device)], dim=1)
            log_p = torch.logaddexp(log_p, prev_next)
            
            # Skip: s -> s+2
            # Shift Left 2
            prev_skip = torch.cat([weighted_beta[:, 2:], torch.full((B, 2), -float('inf'), device=device)], dim=1)
            
            log_p = torch.where(skip_permission_mask, torch.logaddexp(log_p, prev_skip), log_p)
            
            # Update beta for next iteration (which is t-1)
            beta = log_p

    return beta_history

class CTCLossFunction(autograd.Function):
    @staticmethod
    def forward(ctx, log_probs, targets, input_lengths, target_lengths, blank=0, zero_infinity=False):
        # 1. Prep
        # Targets prep (pad) - duplicate logic from previous to ensure self-contained or passed in?
        # Passed in raw flat targets. Need to pad here.
        # Re-use the "Extension" logic.
        
        device = log_probs.device
        T_max, B, C = log_probs.shape
        L_max = target_lengths.max().item()
        S_max = 2 * L_max + 1
        
        extended_targets = torch.full((B, S_max), blank, dtype=torch.long, device=device)
        target_start = 0
        for b in range(B):
            L = target_lengths[b].item()
            current_target = targets[target_start : target_start + L]
            target_start += L
            extended_targets[b, 1:2*L+1:2] = current_target
            
        # 2. Compute Alpha
        alpha = _compute_alpha_matrix(log_probs, extended_targets, input_lengths, target_lengths, blank)
        
        # 3. Compute Loss
        # Gather final loss from alpha at T-1
        losses = []
        for b in range(B):
            T = input_lengths[b].item()
            L = target_lengths[b].item()
            idx1 = 2 * L
            idx2 = 2 * L - 1
            final_alpha = alpha[T-1, b]
            if L > 0:
                total_log_prob = torch.logaddexp(final_alpha[idx1], final_alpha[idx2])
            else:
                total_log_prob = final_alpha[idx1]
            losses.append(-total_log_prob)
        losses = torch.stack(losses)
        
        # Save for backward
        ctx.save_for_backward(log_probs, alpha, extended_targets, input_lengths, target_lengths, losses)
        ctx.blank = blank
        ctx.zero_infinity = zero_infinity
        
        # Handle zero_infinity check for Output
        if zero_infinity:
            inf_mask = torch.isinf(losses) | torch.isnan(losses)
            if inf_mask.any():
                losses = torch.where(inf_mask, torch.zeros_like(losses), losses)

        return losses

    @staticmethod
    def backward(ctx, grad_output):
        log_probs, alpha, extended_targets, input_lengths, target_lengths, neg_log_likelihood = ctx.saved_tensors
        blank = ctx.blank
        zero_infinity = ctx.zero_infinity
        
        # 1. Compute Beta
        beta = _compute_beta_matrix(log_probs, extended_targets, input_lengths, target_lengths, blank)
        
        # 2. Compute Gradient
        # grad = - exp(alpha + beta - total - log_y)
        # where total = -losses (since losses = -total_log_prob)
        # But we need to handle broadcasting.
        
        # Total log prob per batch
        total_log_prob = -neg_log_likelihood # [B]
        
        # Compute joint log prob: alpha + beta
        # [T, B, S]
        joint = alpha + beta
        
        # Normalize by total probability: joint - total[:, None, None] (broadcast)
        # [T, B, S] - [B]
        # Mask out T > input_length to avoid NaNs?
        # Safe to just do it? alpha/beta are -inf outside.
        
        normalized_joint = joint - total_log_prob.view(1, -1, 1)
        
        # Calculate gradients for paths
        # grad = - exp(normalized_joint - log_y_gathered) ?
        # No, formula: dL/d(log_y) = - posterior = - alpha*beta/(P * y) * y ?
        # No, y cancels out.
        # dL/d(logit) = y - posterior_sum.
        # dL/d(log_y) = - posterior_sum.
        # posterior_sum(t, b, k) = sum_{s: char[s]==k} exp(normalized_joint(t, b, s))
        
        # Input log_probs is [T, B, C]
        T_max, B, C = log_probs.shape
        device = log_probs.device
        
        # We need to scatter 'exp(normalized_joint)' back to C class indices
        # extended_targets [B, S] maps S -> Class Index
        
        logits_grad = torch.zeros((T_max, B, C), device=device)
        
        # We can iterate S and add? Or use scatter_add.
        # joint_exp = exp(normalized_joint)
        # But watch out for -inf.
        # Using log_sum_exp equivalents? No, we need standard summation of probabilities.
        
        # Safeguard: if loss was inf (total_log_prob = -inf), then normalized_joint is nan.
        # Zero gradients there.
        valid_loss_mask = ~ (torch.isinf(total_log_prob) | torch.isnan(total_log_prob))
        
        # Apply mask
        normalized_joint[:, ~valid_loss_mask, :] = -float('inf')
        
        posterior_probs = torch.exp(normalized_joint) # [T, B, S]
        
        # Scatter add
        # Target indices [1, B, S] -> [T, B, S]
        target_indices = extended_targets.unsqueeze(0).expand(T_max, -1, -1)
        
        logits_grad.scatter_add_(2, target_indices, posterior_probs)
        
        # This is the "positive" part of the gradient (posterior prob).
        # Since we want dL/d(log_y), and L is negative log likelihood.
        # Gradient is -1 * posterior.
        # So grad_input = -logits_grad.
        
        grad_input = -logits_grad
        
        # Handle grad_output (upstream gradient from e.g. mean reduction)
        # grad_output is [B] (if no reduction) or scalar?
        # If reduction='mean', forward returns scalar. grad_output is scalar.
        # But wait, our forward returns [B] losses.
        # reduction is handled in Module, not Function.
        # So grad_output should be [B].
        
        # Multiply by grad_output
        # grad_input: [T, B, C]
        # grad_output: [B]
        
        grad_input = grad_input * grad_output.view(1, -1, 1)
        
        # Zero out padding gradients (T > input_lengths)
        # Create mask
        # T_indices = arange(T).
        t_indices = torch.arange(T_max, device=device).unsqueeze(1) # [T, 1]
        valid_steps = t_indices < input_lengths.unsqueeze(0) # [T, B]
        
        grad_input = torch.where(valid_steps.unsqueeze(2), grad_input, torch.zeros_like(grad_input))
        
        # Zero Infinity Gradients (redundant check but safe)
        if zero_infinity:
             grad_input = torch.where(torch.isnan(grad_input) | torch.isinf(grad_input), torch.zeros_like(grad_input), grad_input)

        return grad_input, None, None, None, None, None

class CTCLossFromScratch(nn.Module):
    def __init__(self, blank=0, reduction='mean', zero_infinity=False):
        super(CTCLossFromScratch, self).__init__()
        self.blank = blank
        self.reduction = reduction
        self.zero_infinity = zero_infinity

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        # Delegate to Function
        losses = CTCLossFunction.apply(log_probs, targets, input_lengths, target_lengths, self.blank, self.zero_infinity)
        
        if self.reduction == 'mean':
            # Normalize by target length first as per observation?
            # Or just mean of losses?
            # Existing notebook logic said: Python Loss / TargetLen Matches PyTorch
            # So "raw" NLL should be divided by target_lengths.
             safe_lengths = target_lengths.clone().float()
             safe_lengths[safe_lengths == 0] = 1.0
             losses = losses / safe_lengths
             return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:
            return losses
