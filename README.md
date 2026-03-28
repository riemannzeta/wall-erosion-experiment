# Wall Erosion Experiment

Testing the **synchronization tax prediction**: can the Shannon/Kolmogorov "wall" be eroded by providing indirect gradient flow to unrewarded positions?

## Background

[Misra](https://medium.com/@vishalmisra/the-wall-between-shannon-and-kolmogorov-65a9d7e8fb7c) provides a clean demonstration of a fundamental generalization failure: when a transformer is trained on modular linear recurrences (`x_{t+1} = ax_t + b mod 17`) with cross-entropy loss computed only at positions 1-K, the model achieves near-Bayesian precision at trained positions but fails catastrophically at untrained positions. Misra interprets this "wall" as an intrinsic epistemological boundary: LLMs compile localized prediction circuits based on pattern matching (Shannon) rather than learning generalized, position-independent algorithms (Kolmogorov).

This repository tests an alternative interpretation based on the [Maintaining Divergence](https://www.symmetrybroken.com/maintaining-divergence/#the-three-part-decomposition) framework. This thermodynamic view of inference predicts that the wall reflects where training allocates resources—a failure to pay the "synchronization tax" required to maintain coherence out-of-distribution—rather than a strict architectural inability to learn the algorithm.

**The Original Prediction:**

> If the wall simply reflects where synchronization costs are paid, providing a generic "maintenance subsidy" (indirect, non-task-specific gradient flow) to unrewarded positions should provide the continuous computational energy needed to maintain coherence and erode the wall.

**The Findings:**

The experimental data disciplines both frameworks productively. The original prediction was too strong, but the strict Shannon/Kolmogorov framing is too rigid.

The original prediction fails because generic gradient flow — providing compute without task-relevant information — does not erode the wall. Matched controls confirm this cleanly: entropy regularization toward a uniform target, distillation from a random teacher, and hidden-state smoothness constraints all preserve the wall. Misra is right that generic compute is not enough; you cannot pay a synchronization tax with unconstrained kinetic energy.

But the wall is not as thick as the Shannon/Kolmogorov framing implies. Two mechanisms eliminate it completely:

- **Distillation** from a trained teacher erodes the wall by supplying the full Bayesian posterior at unrewarded positions. This supports Misra's interpretation: the teacher explicitly hands the student the missing mathematical beliefs.
- **Entropy regularization** erodes the wall by supplying only a single, label-agnostic scalar signal — *how uncertain to be* — without specifying *what to predict*. Given only this minimal calibration hint, the model flawlessly reconstructs the highly complex Bayesian posterior from its own internal representations. This mathematically challenges a hard Shannon/Kolmogorov divide. If the model were purely a Shannon curve-fitter that never learned the algorithm, forcing it to "be confident" out-of-distribution would simply amplify hallucinated garbage. The fact that it snaps to the exact correct answer proves the generalized Kolmogorov circuit *is* globally compiled in the trained weights.

**Conclusion:**

The experiment disciplines the [Maintaining Divergence](https://www.symmetrybroken.com/maintaining-divergence/) framework: the "synchronization tax" cannot be paid with generic compute. Maintaining an algorithmic channel requires structural alignment, not just gradient flow.

However, the entropy regularization result fundamentally shifts our understanding of the wall. It is not a **compilation barrier** (a lack of algorithmic capability), but a **calibration barrier** caused by environmental entanglement. The model learns the universal rule, but because it lacks an endogenous mechanism to shield that rule from out-of-distribution positional noise, its epistemic confidence collapses at novel positions. The scalar entropy target acts as an exogenous, metaphorical "roof"—providing the minimal maintenance structure needed to protect the computational channel from the environment, allowing the model to confidently deploy the generalized circuit it already possesses. Whether this distinction between an unlearned algorithm and a fragile, environmentally entangled one matters for practical LLM limitations remains an open question.

**Phase 2 Addendum (full convergence, 3 seeds):** At 150K steps the integer wall ratio reaches ~76x (approaching Misra's reported 83x), confirming Phase 1's 7.1x was a diagnostic snapshot. The wall also generalizes to sequence-length extrapolation: models trained on length-8 sequences show a clean wall at position 9 when evaluated on longer sequences. Notably, models that cannot extrapolate revert to maximum entropy (uninformative but honest prediction) rather than hallucinating — the failure mode is calibrated uncertainty, not confabulation.

## Results

The wall **is not intrinsic**. Two mechanisms completely eliminate it:

| Condition | Trained MAE | Untrained MAE | Wall Ratio |
|-----------|------------|---------------|------------|
| Baseline-Horizon (the wall) | 0.247 | 1.755 | **7.1x** |
| A: Entropy regularization | 0.390 | **0.272** | **0.7x** |
| A: Entropy control (uniform) | 0.248 | 2.092 | 8.4x |
| B: Soft distillation | 0.225 | **0.045** | **0.2x** |
| B: Distill control (random) | 0.185 | 2.084 | 11.3x |

Matched controls that provide gradient flow but no task-relevant information preserve the wall, confirming the effect is driven by *information content*, not gradient flow alone.

Full results: [`results/RESULTS.md`](results/RESULTS.md)

### Phase 2: Extrapolation (full convergence, 3 seeds)

| Condition | Trained MAE | Untrained MAE | Wall Ratio |
|-----------|------------|---------------|------------|
| Horizon — Integer | 0.020 | 1.512 | **~76x** |
| Horizon — Opaque | 0.824 | 1.796 | **~2.2x** |

The wall at convergence approaches Misra's reported 83x. Opaque tokens reduce the ratio but by degrading trained performance, not improving untrained.

Extrapolation mode reveals a **second wall**: models trained on 8-position sequences generalize perfectly within that range but fail cleanly at position 9+, reverting to maximum entropy rather than hallucinating.

Full results: [`results/RESULTS.md`](results/RESULTS.md)

### Per-position MAE

![Per-position MAE curves](figures/wall_erosion_per_position.png)

## Mechanisms tested

| Mechanism | What it provides at unrewarded positions | Wall erosion |
|-----------|----------------------------------------|-------------|
| **A: Entropy reg.** | Target entropy (how uncertain to be) | Complete |
| **B: Distillation** | Soft output distribution from trained teacher | Complete |
| **C: Smoothness** | Hidden-state continuity constraint | None (regularization artifact) |
| **D: Aux classifier** | Binary "is this a program?" signal | Modest |

Each mechanism has a matched control providing gradient flow with no task-relevant information.

## Reproducing

```bash
# Setup (uv preferred)
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Or with pip:
# python -m venv .venv && source .venv/bin/activate
# pip install -r requirements.txt

# --- Phase 1: Wall erosion mechanisms (10K-step diagnostic) ---

# Reproduce baselines (use --device mps on Apple Silicon, --device cuda on NVIDIA)
python wall_erosion_experiment.py --mechanism none --loss_horizon 15 \
    --n_steps 10000 --eval_every 5000 --device mps --seeds 42

python wall_erosion_experiment.py --mechanism none \
    --n_steps 10000 --eval_every 5000 --device mps --seeds 42

# Train teacher (for distillation)
python wall_erosion_experiment.py --train_teacher \
    --n_steps 10000 --eval_every 5000 --device mps --seeds 42

# Run a mechanism
python wall_erosion_experiment.py --mechanism entropy --subsidy_lambda 0.1 \
    --n_steps 10000 --eval_every 5000 --device mps --seeds 42

# Run with control
python wall_erosion_experiment.py --mechanism entropy --control --subsidy_lambda 0.1 \
    --n_steps 10000 --eval_every 5000 --device mps --seeds 42

# Full matrix (all mechanisms x controls x lambda sweep x 3 seeds)
python wall_erosion_experiment.py --run_matrix --seeds 42 43 44 --device mps

# --- Phase 2: Extrapolation experiments (full convergence, ~2h per run on MPS) ---

# Horizon mode — integer and opaque
python recurrence_extrapolation.py --mode horizon --seeds 42 43 44 --device mps
python recurrence_extrapolation.py --mode horizon --opaque --seeds 42 43 44 --device mps

# Extrapolation mode — integer and opaque
python recurrence_extrapolation.py --mode extrapolate --train_seq_len 8 \
    --eval_seq_lens 8 16 32 50 --sinusoidal_pe --seeds 42 43 44 --device mps
python recurrence_extrapolation.py --mode extrapolate --opaque --train_seq_len 8 \
    --eval_seq_lens 8 16 32 50 --sinusoidal_pe --seeds 42 43 44 --device mps
```

## Upstream

The base task (modular linear recurrence wind tunnel) is from [vishalmisra/bayesian-wind-tunnel](https://github.com/vishalmisra/bayesian-wind-tunnel). Files `recurrence_bwt.py` and `recurrence_extrapolation.py` are from that repo and provide data generation, Bayesian ground truth computation, and evaluation.

## License

This experiment code is released under the MIT License. The upstream files (`recurrence_bwt.py`, `recurrence_extrapolation.py`) are from [vishalmisra/bayesian-wind-tunnel](https://github.com/vishalmisra/bayesian-wind-tunnel) and are subject to its license terms.
