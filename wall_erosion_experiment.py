"""
Wall Erosion Experiment — Testing the Synchronization Tax Prediction
====================================================================

Tests whether the Shannon/Kolmogorov "wall" (Misra 2025) can be eroded
by providing indirect gradient flow to unrewarded positions.

The Maintaining Divergence framework predicts: the wall is not intrinsic
to gradient descent but reflects where synchronization costs are paid.
Any mechanism providing even indirect gradient to unrewarded positions
should soften the wall in proportion to the subsidy.

Four synchronization subsidy mechanisms:

  A) Entropy regularization — match Bayesian entropy at unrewarded positions
  B) Soft distillation — KL from a fully-trained teacher at unrewarded positions
  C) Representation smoothness — penalize hidden-state jumps past the horizon
  D) Auxiliary classifier — predict P(H_P) at every position

Each mechanism has a matched control that provides gradient flow but
no task-relevant information, isolating the information effect from
the regularization effect.

Usage:
    # Reproduce baseline wall
    python wall_erosion_experiment.py --mechanism none --seeds 42 43 44

    # Entropy regularization with lambda sweep
    python wall_erosion_experiment.py --mechanism entropy --subsidy_lambda 0.1 --seeds 42 43 44

    # Entropy control (uniform target)
    python wall_erosion_experiment.py --mechanism entropy --control --subsidy_lambda 0.1 --seeds 42 43 44

    # Train teacher for distillation
    python wall_erosion_experiment.py --train_teacher --seeds 42 43 44

    # Distillation from teacher
    python wall_erosion_experiment.py --mechanism distill --subsidy_lambda 0.5 \
        --teacher_checkpoint results/wall_erosion/teacher_seed42/best_model.pt --seeds 42

    # Run full matrix
    python wall_erosion_experiment.py --run_matrix --seeds 42 43 44
"""

import os
import math
import argparse
import json
import numpy as np

from recurrence_bwt import (
    RecurrenceConfig,
    generate_recurrence_sequence,
    bayesian_predictive_recurrence,
    class_posterior_recurrence,
    count_consistent_recurrences,
    _predictive_entropy,
)
from recurrence_extrapolation import evaluate_at_length

# Lazy torch imports
torch = None
nn = None
F = None


def _ensure_torch():
    global torch, nn, F
    if torch is None:
        import torch as _torch
        import torch.nn as _nn
        import torch.nn.functional as _F
        torch = _torch
        nn = _nn
        F = _F


def _resolve_device(requested):
    """Resolve device string, with MPS fallback for Apple Silicon."""
    _ensure_torch()
    if requested == 'cuda' and not torch.cuda.is_available():
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    return torch.device(requested)


def _seed_all(seed):
    """Seed all RNGs."""
    _ensure_torch()
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


# ============================================================================
# Model with hidden state access and optional auxiliary head
# ============================================================================

def _build_model_class():
    _ensure_torch()

    class MultiHeadAttention(nn.Module):
        def __init__(self, d_model, n_heads, dropout=0.1):
            super().__init__()
            self.n_heads = n_heads
            self.d_head = d_model // n_heads
            self.qkv = nn.Linear(d_model, 3 * d_model)
            self.out_proj = nn.Linear(d_model, d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, mask=None):
            B, T, C = x.shape
            qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
            q, k, v = qkv.unbind(dim=2)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            scale = self.d_head ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            if mask is not None:
                attn = attn.masked_fill(
                    mask.unsqueeze(0).unsqueeze(0), float('-inf')
                )
            alpha = torch.softmax(attn, dim=-1)
            alpha = self.dropout(alpha)
            out = (alpha @ v).transpose(1, 2).reshape(B, T, C)
            return self.out_proj(out), alpha

    class TransformerBlock(nn.Module):
        def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1):
            super().__init__()
            d_ff = d_ff or 4 * d_model
            self.attn = MultiHeadAttention(d_model, n_heads, dropout)
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.ff = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            )

        def forward(self, x, mask=None):
            h, alpha = self.attn(self.ln1(x), mask)
            x = x + h
            x = x + self.ff(self.ln2(x))
            return x, alpha

    class RecurrenceTransformerSubsidy(nn.Module):
        """Transformer that returns both logits and hidden states.

        Architecture identical to RecurrenceTransformerExtrap (learned PE).
        Adds forward_with_hiddens() for subsidy loss computation and
        an optional auxiliary classification head.
        """

        def __init__(self, vocab_size, n_tokens, d_model=192, n_layers=6,
                     n_heads=6, d_ff=768, dropout=0.1, aux_classifier=False):
            super().__init__()
            self.vocab_size = vocab_size
            self.n_tokens = n_tokens
            self.d_model = d_model

            self.token_embed = nn.Embedding(
                vocab_size + 1, d_model, padding_idx=vocab_size
            )
            self.pos_embed = nn.Embedding(512, d_model)

            self.layers = nn.ModuleList([
                TransformerBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ])

            self.ln_final = nn.LayerNorm(d_model)
            self.output_proj = nn.Linear(d_model, n_tokens)

            # Optional auxiliary classifier for mechanism D
            self.aux_head = None
            if aux_classifier:
                self.aux_head = nn.Linear(d_model, 1)

        def _encode(self, tokens):
            """Shared encoding up to final layer norm. Returns hidden states."""
            B, T = tokens.shape
            mask = torch.triu(
                torch.ones(T, T, device=tokens.device), diagonal=1
            ).bool()

            x = self.token_embed(tokens)
            positions = torch.arange(
                T, device=tokens.device
            ).unsqueeze(0).expand(B, -1)
            x = x + self.pos_embed(positions)

            for layer in self.layers:
                x, _ = layer(x, mask)

            hiddens = self.ln_final(x)
            return hiddens

        def forward(self, tokens):
            """Standard forward — returns logits only (compatible with eval)."""
            hiddens = self._encode(tokens)
            return self.output_proj(hiddens)

        def forward_with_hiddens(self, tokens):
            """Returns (logits, hiddens, aux_logits).

            hiddens: (B, T, d_model) after ln_final, before output_proj
            aux_logits: (B, T, 1) if aux_head exists, else None
            """
            hiddens = self._encode(tokens)
            logits = self.output_proj(hiddens)
            aux_logits = None
            if self.aux_head is not None:
                aux_logits = self.aux_head(hiddens)
            return logits, hiddens, aux_logits

    return RecurrenceTransformerSubsidy


# ============================================================================
# Vectorized loss computations
# ============================================================================

def _masked_ce_loss(logits, targets, mask, n_tokens):
    """Vectorized cross-entropy over masked positions.

    Args:
        logits: (B, T, n_tokens)
        targets: (B, T) long tensor of target token ids
        mask: (B, T) bool tensor — True where loss should be computed
        n_tokens: number of classes
    Returns:
        Scalar mean CE loss over masked positions
    """
    if not mask.any():
        return torch.tensor(0.0, device=logits.device)

    # Flatten masked positions
    logits_flat = logits[mask]  # (N, n_tokens)
    targets_flat = targets[mask]  # (N,)
    return F.cross_entropy(logits_flat, targets_flat)


def _entropy_from_logits(logits):
    """Compute entropy in bits from logits. Shape-preserving.

    Args:
        logits: (..., n_tokens)
    Returns:
        entropy: (...) in bits
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    # H = -sum(p * log2(p)) = -sum(p * ln(p)) / ln(2)
    return -(probs * log_probs).sum(dim=-1) / math.log(2)


def compute_entropy_subsidy(logits, entropy_targets, unrewarded_mask, n_tokens):
    """Mechanism A: MSE between model entropy and target entropy.

    Args:
        logits: (B, T, n_tokens)
        entropy_targets: (B, T) float tensor of target entropies in bits
        unrewarded_mask: (B, T) bool tensor — True at unrewarded positions
        n_tokens: number of classes
    Returns:
        Scalar MSE loss
    """
    if not unrewarded_mask.any():
        return torch.tensor(0.0, device=logits.device)

    H_model = _entropy_from_logits(logits[:, :, :n_tokens])  # (B, T)
    diff = H_model[unrewarded_mask] - entropy_targets[unrewarded_mask]
    return (diff ** 2).mean()


def compute_distill_subsidy(student_logits, teacher_logits, unrewarded_mask,
                            n_tokens, temperature=2.0):
    """Mechanism B: KL divergence from teacher at unrewarded positions.

    Args:
        student_logits: (B, T, n_tokens)
        teacher_logits: (B, T, n_tokens) from frozen teacher
        unrewarded_mask: (B, T) bool tensor
        n_tokens: number of classes
        temperature: softmax temperature
    Returns:
        Scalar T^2 * mean KL loss
    """
    if not unrewarded_mask.any():
        return torch.tensor(0.0, device=student_logits.device)

    s = student_logits[:, :, :n_tokens][unrewarded_mask] / temperature
    t = teacher_logits[:, :, :n_tokens][unrewarded_mask] / temperature

    s_log_probs = F.log_softmax(s, dim=-1)
    t_probs = F.softmax(t, dim=-1)

    kl = F.kl_div(s_log_probs, t_probs, reduction='batchmean')
    return (temperature ** 2) * kl


def compute_smooth_subsidy(hiddens, smooth_mask):
    """Mechanism C: Penalize hidden-state jumps at masked positions.

    Args:
        hiddens: (B, T, d_model)
        smooth_mask: (B, T) bool tensor — True where smoothness is enforced
            (consecutive diff h[t] - h[t-1] is penalized where smooth_mask[t] is True)
    Returns:
        Scalar mean squared diff
    """
    if not smooth_mask[:, 1:].any():
        return torch.tensor(0.0, device=hiddens.device)

    # Consecutive diffs: h[t] - h[t-1] for t=1..T-1
    diffs = hiddens[:, 1:, :] - hiddens[:, :-1, :]  # (B, T-1, d_model)
    mask = smooth_mask[:, 1:]  # (B, T-1) — shifted to align with diffs

    if not mask.any():
        return torch.tensor(0.0, device=hiddens.device)

    masked_diffs = diffs[mask]  # (N, d_model)
    return (masked_diffs ** 2).mean()


def compute_classify_subsidy(aux_logits, labels, valid_mask):
    """Mechanism D: BCE for auxiliary program classifier at all positions.

    Args:
        aux_logits: (B, T, 1)
        labels: (B,) float tensor — 1.0 for program, 0.0 for random
        valid_mask: (B, T) bool tensor — True at valid positions
    Returns:
        Scalar BCE loss
    """
    if aux_logits is None or not valid_mask.any():
        return torch.tensor(0.0, device=labels.device)

    # Expand labels to match positions: (B,) -> (B, T)
    labels_expanded = labels.unsqueeze(1).expand_as(valid_mask).float()

    pred = aux_logits[:, :, 0][valid_mask]  # (N,)
    target = labels_expanded[valid_mask]  # (N,)
    return F.binary_cross_entropy_with_logits(pred, target)


# ============================================================================
# Wall metrics
# ============================================================================

def compute_wall_metrics(per_pos, loss_horizon, train_seq_len):
    """Compute wall ratio, sharpness, and per-region MAE."""
    trained_maes = [
        per_pos[t]['mae_mean'] for t in per_pos if 1 <= t <= loss_horizon
    ]
    untrained_maes = [
        per_pos[t]['mae_mean'] for t in per_pos if t > loss_horizon
    ]

    trained_mae = float(np.mean(trained_maes)) if trained_maes else 0.0
    untrained_mae = float(np.mean(untrained_maes)) if untrained_maes else 0.0

    wall_ratio = untrained_mae / trained_mae if trained_mae > 0 else float('inf')

    wall_sharpness = 0.0
    if loss_horizon in per_pos and (loss_horizon + 1) in per_pos:
        wall_sharpness = (per_pos[loss_horizon + 1]['mae_mean']
                          - per_pos[loss_horizon]['mae_mean'])

    return {
        'trained_mae': trained_mae,
        'untrained_mae': untrained_mae,
        'wall_ratio': wall_ratio,
        'wall_sharpness': wall_sharpness,
    }


def compute_erosion_fraction(wall_ratio, wr_full=1.0, wr_horizon=83.0):
    """How far between the wall (EF=0) and full supervision (EF=1)."""
    if wr_horizon <= wr_full:
        return 0.0
    return 1.0 - (wall_ratio - wr_full) / (wr_horizon - wr_full)


# ============================================================================
# Batch generation helper
# ============================================================================

def generate_batch(p, pi, seq_len, batch_size, device):
    """Generate a batch of sequences with ground truth metadata.

    Returns:
        x: (B, seq_len) long tensor of tokens on device
        entropy_targets: (B, seq_len) float tensor of Bayesian entropy at each position
        is_program: (B,) float tensor — 1.0 if program, 0.0 if random
    """
    _ensure_torch()
    cfg = RecurrenceConfig(p=p, pi=pi, seq_len=seq_len)

    all_tokens = []
    all_entropies = []
    all_is_program = []

    for _ in range(batch_size):
        tokens, gt, metadata = generate_recurrence_sequence(cfg)
        all_tokens.append(tokens)

        # Extract per-position entropy from ground truth
        entropies = [0.0] * seq_len  # position 0 has no prediction
        for entry in gt:
            t = entry['t']
            if 0 <= t < seq_len:
                entropies[t] = entry['entropy']
        all_entropies.append(entropies)

        all_is_program.append(1.0 if metadata['true_class'] == 'program' else 0.0)

    x = torch.tensor(all_tokens, dtype=torch.long).to(device)
    entropy_targets = torch.tensor(all_entropies, dtype=torch.float32).to(device)
    is_program = torch.tensor(all_is_program, dtype=torch.float32).to(device)

    return x, entropy_targets, is_program


# ============================================================================
# Training
# ============================================================================

def train(args):
    _ensure_torch()
    RecurrenceTransformerSubsidy = _build_model_class()

    device = _resolve_device(args.device)
    p = args.p
    vocab_size = p
    n_tokens = p
    loss_horizon = args.loss_horizon
    train_seq_len = args.train_seq_len
    mechanism = args.mechanism

    use_aux = (mechanism == 'classify')
    model = RecurrenceTransformerSubsidy(
        vocab_size=vocab_size,
        n_tokens=n_tokens,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        aux_classifier=use_aux,
    ).to(device)

    param_count = sum(pr.numel() for pr in model.parameters())
    print(f"Model: {param_count:,} parameters on {device}")
    print(f"Mechanism: {mechanism} (control={args.control})")
    print(f"Subsidy lambda: {args.subsidy_lambda}")
    print(f"Loss horizon: 1-{loss_horizon} (of {train_seq_len})")

    # Load teacher for distillation
    teacher = None
    if mechanism == 'distill':
        teacher = RecurrenceTransformerSubsidy(
            vocab_size=vocab_size, n_tokens=n_tokens,
            d_model=args.d_model, n_layers=args.n_layers,
            n_heads=args.n_heads, d_ff=args.d_ff,
            dropout=0.0, aux_classifier=False,
        ).to(device)

        if args.control:
            print("  Teacher: RANDOM (untrained) — control condition")
        else:
            if not args.teacher_checkpoint:
                raise ValueError(
                    "Distillation requires --teacher_checkpoint "
                    "(or --control for random teacher)"
                )
            state = torch.load(args.teacher_checkpoint, map_location=device,
                               weights_only=True)
            teacher.load_state_dict(state)
            print(f"  Teacher loaded from {args.teacher_checkpoint}")

        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_steps
    )

    os.makedirs(args.output_dir, exist_ok=True)
    best_mae = float('inf')
    losses_ce = []
    losses_sub = []

    # Pre-compute masks (same for all batches since seq_len is fixed, no opaque)
    # Positions: 0, 1, ..., seq_len-1
    # Predict token at position t+1 from logits at position t
    # So loss at position t means: logits[t] predicts x[t+1]
    # Rewarded: positions 0..loss_horizon-1 (predicting tokens 1..loss_horizon)
    # Unrewarded: positions loss_horizon..seq_len-2 (predicting tokens loss_horizon+1..seq_len-1)
    rewarded_mask = torch.zeros(1, train_seq_len, dtype=torch.bool, device=device)
    rewarded_mask[0, :loss_horizon] = True  # positions 0..K-1

    unrewarded_mask = torch.zeros(1, train_seq_len, dtype=torch.bool, device=device)
    unrewarded_mask[0, loss_horizon:train_seq_len - 1] = True  # positions K..T-2

    all_valid_mask = torch.zeros(1, train_seq_len, dtype=torch.bool, device=device)
    all_valid_mask[0, :train_seq_len - 1] = True  # positions 0..T-2

    # Smooth mask for mechanism C: enforce smoothness at unrewarded positions
    # For control: only within rewarded positions
    if mechanism == 'smooth' and args.control:
        smooth_mask_tmpl = torch.zeros(1, train_seq_len, dtype=torch.bool, device=device)
        smooth_mask_tmpl[0, 1:loss_horizon] = True  # within rewarded
    else:
        smooth_mask_tmpl = unrewarded_mask.clone()

    for step in range(1, args.n_steps + 1):
        model.train()

        # Generate batch (vectorized output)
        x, entropy_targets, is_program = generate_batch(
            p, args.pi, train_seq_len, args.batch_size, device
        )
        B = x.shape[0]

        # Expand masks to batch size
        rew_mask = rewarded_mask.expand(B, -1)
        unrew_mask = unrewarded_mask.expand(B, -1)
        valid_mask = all_valid_mask.expand(B, -1)
        s_mask = smooth_mask_tmpl.expand(B, -1)

        # Targets: shifted by 1 (predict next token)
        targets = x[:, 1:]  # (B, T-1)

        # Forward pass
        need_hiddens = mechanism in ('smooth', 'classify', 'entropy', 'distill')
        if need_hiddens:
            logits, hiddens, aux_logits = model.forward_with_hiddens(x)
        else:
            logits = model(x)
            hiddens = None
            aux_logits = None

        # Logits for prediction: positions 0..T-2 predict tokens 1..T-1
        pred_logits = logits[:, :-1, :n_tokens]  # (B, T-1, n_tokens)

        # === CE loss at rewarded positions ===
        ce_mask = rew_mask[:, :-1]  # (B, T-1) — align with pred_logits
        ce_loss = _masked_ce_loss(pred_logits, targets, ce_mask, n_tokens)

        # === Subsidy loss ===
        sub_loss = torch.tensor(0.0, device=device)

        if mechanism != 'none' and args.subsidy_lambda > 0:
            if mechanism == 'entropy':
                # Entropy targets at prediction positions
                # Position t in logits predicts t+1, entropy target for prediction
                # at position t is the entropy of P(x_{t+1} | x_0..x_t)
                # which is stored at gt[t+1]['entropy'] (= entropy_targets[:, t+1])
                # So for pred_logits at index t (=position t), target = entropy_targets[:, t+1]
                ent_targets_shifted = entropy_targets[:, 1:]  # (B, T-1)
                unrew_shifted = unrew_mask[:, :-1]  # (B, T-1)

                if args.control:
                    # Control: uniform entropy
                    ent_targets_shifted = torch.full_like(
                        ent_targets_shifted, math.log2(p)
                    )

                sub_loss = compute_entropy_subsidy(
                    logits[:, :-1, :], ent_targets_shifted, unrew_shifted, n_tokens
                )

            elif mechanism == 'distill':
                with torch.no_grad():
                    teacher_logits = teacher(x)
                unrew_shifted = unrew_mask[:, :-1]
                sub_loss = compute_distill_subsidy(
                    logits[:, :-1, :], teacher_logits[:, :-1, :],
                    unrew_shifted, n_tokens, temperature=2.0
                )

            elif mechanism == 'smooth':
                sub_loss = compute_smooth_subsidy(hiddens, s_mask)

            elif mechanism == 'classify':
                labels = is_program
                if args.control:
                    labels = torch.bernoulli(
                        torch.full((B,), 0.5, device=device)
                    )
                sub_loss = compute_classify_subsidy(
                    aux_logits, labels, valid_mask
                )

        # === Total loss ===
        total_loss = ce_loss + args.subsidy_lambda * sub_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses_ce.append(ce_loss.item())
        losses_sub.append(sub_loss.item())

        if step % args.log_every == 0:
            recent_ce = np.mean(losses_ce[-args.log_every:])
            recent_sub = np.mean(losses_sub[-args.log_every:])
            print(f"  Step {step}/{args.n_steps}: CE={recent_ce:.4f}, "
                  f"sub={recent_sub:.4f}", flush=True)

        if step % args.eval_every == 0:
            metrics, per_pos = evaluate_at_length(
                model, p, args.pi, train_seq_len,
                n_eval=500, device=str(device)
            )
            wm = compute_wall_metrics(per_pos, loss_horizon, train_seq_len)

            print(f"  Eval: MAE={metrics['mae_bits']:.4f}, "
                  f"WR={wm['wall_ratio']:.1f}x, "
                  f"trained={wm['trained_mae']:.4f}, "
                  f"untrained={wm['untrained_mae']:.4f}", flush=True)

            for t in sorted(per_pos.keys()):
                pp = per_pos[t]
                marker = " [T]" if t <= loss_horizon else " [U]"
                print(f"    t={t:2d}: MAE={pp['mae_mean']:.4f}{marker}")

            if metrics['mae_bits'] < best_mae:
                best_mae = metrics['mae_bits']
                torch.save(
                    model.state_dict(),
                    os.path.join(args.output_dir, 'best_model.pt')
                )

    # ====================================================================
    # Final evaluation
    # ====================================================================
    print(f"\n{'='*70}", flush=True)
    print("FINAL EVALUATION")
    print(f"{'='*70}")

    metrics, per_pos = evaluate_at_length(
        model, p, args.pi, train_seq_len,
        n_eval=2000, device=str(device)
    )
    wm = compute_wall_metrics(per_pos, loss_horizon, train_seq_len)
    ef = compute_erosion_fraction(wm['wall_ratio'])

    print(f"\n  Mechanism: {mechanism} "
          f"({'control' if args.control else 'active'})")
    print(f"  Lambda: {args.subsidy_lambda}")
    print(f"  Loss horizon: 1-{loss_horizon}")
    print()

    for t in sorted(per_pos.keys()):
        pp = per_pos[t]
        marker = " [TRAINED]" if t <= loss_horizon else " [UNTRAINED]"
        print(f"    t={t:2d}: H_model={pp['H_model_mean']:.4f}, "
              f"H_bayes={pp['H_bayes_mean']:.4f}, "
              f"MAE={pp['mae_mean']:.4f}{marker}")

    print(f"\n  Trained MAE  (1-{loss_horizon}): {wm['trained_mae']:.4f} bits")
    print(f"  Untrained MAE ({loss_horizon+1}-{train_seq_len-1}): "
          f"{wm['untrained_mae']:.4f} bits")
    print(f"  Wall Ratio: {wm['wall_ratio']:.2f}x")
    print(f"  Wall Sharpness: {wm['wall_sharpness']:.4f} bits")
    print(f"  Erosion Fraction: {ef:.4f}")

    results = {
        'mechanism': mechanism,
        'control': args.control,
        'subsidy_lambda': args.subsidy_lambda,
        'loss_horizon': loss_horizon,
        'train_seq_len': train_seq_len,
        'metrics': metrics,
        'per_position': {str(k): v for k, v in per_pos.items()},
        'wall_metrics': wm,
        'erosion_fraction': ef,
    }

    results_path = os.path.join(args.output_dir, 'wall_erosion_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nResults saved to {results_path}", flush=True)

    return results


# ============================================================================
# Teacher training (full-horizon baseline for mechanism B)
# ============================================================================

def train_teacher(args):
    """Train a full-horizon model (CE at all positions). Used as teacher."""
    _ensure_torch()
    RecurrenceTransformerSubsidy = _build_model_class()

    device = _resolve_device(args.device)
    p = args.p
    vocab_size = p
    n_tokens = p

    model = RecurrenceTransformerSubsidy(
        vocab_size=vocab_size, n_tokens=n_tokens,
        d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, d_ff=args.d_ff,
        dropout=args.dropout, aux_classifier=False,
    ).to(device)

    print(f"Training TEACHER (full-horizon CE at all positions)")
    print(f"  {sum(pr.numel() for pr in model.parameters()):,} params on {device}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_steps
    )

    os.makedirs(args.output_dir, exist_ok=True)
    best_mae = float('inf')
    losses = []

    # Full-horizon mask: all positions 0..T-2 predict tokens 1..T-1
    full_mask = torch.ones(1, args.train_seq_len - 1, dtype=torch.bool, device=device)

    for step in range(1, args.n_steps + 1):
        model.train()

        x, _, _ = generate_batch(
            p, args.pi, args.train_seq_len, args.batch_size, device
        )
        B = x.shape[0]

        logits = model(x)
        pred_logits = logits[:, :-1, :n_tokens]
        targets = x[:, 1:]
        mask = full_mask.expand(B, -1)

        loss = _masked_ce_loss(pred_logits, targets, mask, n_tokens)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

        if step % args.log_every == 0:
            print(f"  Step {step}/{args.n_steps}: "
                  f"loss={np.mean(losses[-args.log_every:]):.4f}", flush=True)

        if step % args.eval_every == 0:
            metrics, per_pos = evaluate_at_length(
                model, p, args.pi, args.train_seq_len,
                n_eval=500, device=str(device)
            )
            print(f"  Eval: MAE={metrics['mae_bits']:.4f} bits")
            if metrics['mae_bits'] < best_mae:
                best_mae = metrics['mae_bits']
                torch.save(
                    model.state_dict(),
                    os.path.join(args.output_dir, 'best_model.pt')
                )
                print(f"    New best (MAE={best_mae:.6f})")

    # Final eval
    metrics, per_pos = evaluate_at_length(
        model, p, args.pi, args.train_seq_len,
        n_eval=2000, device=str(device)
    )
    print(f"\nTeacher final MAE: {metrics['mae_bits']:.4f} bits")
    for t in sorted(per_pos.keys()):
        pp = per_pos[t]
        print(f"    t={t:2d}: MAE={pp['mae_mean']:.4f}")

    ckpt_path = os.path.join(args.output_dir, 'best_model.pt')
    print(f"Teacher saved to {ckpt_path}")
    return ckpt_path


# ============================================================================
# Full experimental matrix
# ============================================================================

def run_matrix(args):
    """Run the complete experimental matrix across mechanisms and lambdas."""
    base_output = args.output_dir
    all_results = []

    conditions = [
        # (mechanism, control, lambdas)
        ('none', False, [0.0]),                        # Baseline horizon
        ('entropy', False, [0.01, 0.1, 1.0]),          # A: entropy reg
        ('entropy', True, [0.1]),                       # A-ctrl: uniform entropy
        ('distill', False, [0.1, 0.5, 1.0]),            # B: distillation
        ('distill', True, [0.5]),                        # B-ctrl: random teacher
        ('smooth', False, [0.01, 0.1, 1.0]),            # C: rep smoothness
        ('smooth', True, [0.1]),                         # C-ctrl: rewarded only
        ('classify', False, [0.01, 0.1, 1.0]),          # D: aux classifier
        ('classify', True, [0.1]),                       # D-ctrl: random labels
    ]

    for seed in args.seeds:
        print(f"\n{'#'*70}")
        print(f"# SEED {seed}")
        print(f"{'#'*70}")

        _seed_all(seed)

        # Train teacher for this seed (used by distill conditions)
        teacher_dir = os.path.join(base_output, f'teacher_seed{seed}')
        teacher_args = argparse.Namespace(**vars(args))
        teacher_args.output_dir = teacher_dir
        teacher_ckpt = train_teacher(teacher_args)

        # Also run full-horizon baseline
        _seed_all(seed)
        full_args = argparse.Namespace(**vars(args))
        full_args.mechanism = 'none'
        full_args.subsidy_lambda = 0.0
        full_args.control = False
        full_args.loss_horizon = args.train_seq_len - 1
        full_dir = os.path.join(base_output, f'baseline_full_seed{seed}')
        full_args.output_dir = full_dir

        result = train(full_args)
        result['condition'] = 'baseline_full'
        result['seed'] = seed
        all_results.append(result)

        for mechanism, control, lambdas in conditions:
            for lam in lambdas:
                ctrl_str = "ctrl" if control else "active"
                cond_name = f"{mechanism}_{ctrl_str}_lam{lam}_seed{seed}"
                cond_dir = os.path.join(base_output, cond_name)

                print(f"\n{'='*70}")
                print(f"CONDITION: {cond_name}")
                print(f"{'='*70}")

                _seed_all(seed)

                cond_args = argparse.Namespace(**vars(args))
                cond_args.mechanism = mechanism
                cond_args.control = control
                cond_args.subsidy_lambda = lam
                cond_args.output_dir = cond_dir
                cond_args.teacher_checkpoint = teacher_ckpt
                cond_args.loss_horizon = args.loss_horizon

                result = train(cond_args)
                result['condition'] = cond_name
                result['seed'] = seed
                all_results.append(result)

    # Save summary
    summary_path = os.path.join(base_output, 'wall_erosion_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nFull summary saved to {summary_path}")

    # Print summary table
    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"{'Condition':<45} {'WR':>8} {'EF':>8} {'Trained':>10} {'Untrained':>10}")
    print("-" * 85)
    for r in all_results:
        wm = r['wall_metrics']
        print(f"{r['condition']:<45} {wm['wall_ratio']:>8.2f} "
              f"{r['erosion_fraction']:>8.4f} "
              f"{wm['trained_mae']:>10.4f} {wm['untrained_mae']:>10.4f}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Wall erosion experiment — synchronization tax test')

    # Mode
    parser.add_argument('--train_teacher', action='store_true',
                        help='Train a full-horizon teacher model only')
    parser.add_argument('--run_matrix', action='store_true',
                        help='Run the full experimental matrix')

    # Mechanism
    parser.add_argument('--mechanism',
                        choices=['none', 'entropy', 'distill', 'smooth', 'classify'],
                        default='none',
                        help='Synchronization subsidy mechanism')
    parser.add_argument('--subsidy_lambda', type=float, default=0.0,
                        help='Weight for subsidy loss')
    parser.add_argument('--control', action='store_true',
                        help='Use control variant (gradient flow, no information)')
    parser.add_argument('--teacher_checkpoint', type=str, default=None,
                        help='Path to teacher model checkpoint (for distill)')

    # Task parameters (match recurrence_extrapolation.py defaults)
    parser.add_argument('--p', type=int, default=17)
    parser.add_argument('--pi', type=float, default=0.5)
    parser.add_argument('--train_seq_len', type=int, default=16)
    parser.add_argument('--loss_horizon', type=int, default=5)

    # Architecture
    parser.add_argument('--d_model', type=int, default=192)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--n_heads', type=int, default=6)
    parser.add_argument('--d_ff', type=int, default=768)
    parser.add_argument('--dropout', type=float, default=0.1)

    # Training
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_steps', type=int, default=150000)
    parser.add_argument('--eval_every', type=int, default=10000)
    parser.add_argument('--log_every', type=int, default=500)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--output_dir', type=str,
                        default='results/wall_erosion')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    if args.train_teacher:
        for seed in args.seeds:
            print(f"\n{'='*70}")
            print(f"TRAINING TEACHER, SEED {seed}")
            print(f"{'='*70}")
            _seed_all(seed)
            teacher_args = argparse.Namespace(**vars(args))
            teacher_args.output_dir = os.path.join(
                args.output_dir, f'teacher_seed{seed}'
            )
            train_teacher(teacher_args)
        return

    if args.run_matrix:
        run_matrix(args)
        return

    # Single condition
    for seed in args.seeds:
        print(f"\n{'='*70}")
        print(f"WALL EROSION: mechanism={args.mechanism}, "
              f"lambda={args.subsidy_lambda}, "
              f"control={args.control}, seed={seed}")
        print(f"{'='*70}")

        _seed_all(seed)

        ctrl_str = "ctrl" if args.control else "active"
        seed_dir = os.path.join(
            args.output_dir,
            f'{args.mechanism}_{ctrl_str}_lam{args.subsidy_lambda}_seed{seed}'
        )
        seed_args = argparse.Namespace(**vars(args))
        seed_args.output_dir = seed_dir
        train(seed_args)


if __name__ == '__main__':
    main()
