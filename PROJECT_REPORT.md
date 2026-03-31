# Neural Cryptanalysis — Project Report

## Overview

This project implements a complete neural cryptanalysis framework that trains machine learning models to distinguish reduced-round block ciphers from random permutations. The approach is inspired by Gohr (2019) and the broader field of differential neural cryptanalysis. Given a pair of ciphertexts `(C, C')` produced from plaintexts with a fixed XOR difference `ΔP`, the models learn to predict whether the outputs came from a real cipher or a random permutation.

---

## Contributors

| Name | Role |
|------|------|
| Project Team | Cipher implementations, ML models, experiments, representations |

---

## Libraries Used

| Library | Version | Purpose |
|---------|---------|---------|
| Python | 3.10+ | Core language |
| PyTorch | ≥ 2.0 | Neural network training and inference |
| NumPy | ≥ 1.24 | Array operations, dataset generation |
| tqdm | ≥ 4.0 | Progress bars during dataset generation |
| matplotlib | ≥ 3.7 | Plot generation (accuracy vs rounds/rep/model) |

Install all dependencies:
```bash
pip install torch numpy tqdm matplotlib
```

---

## Problem Statement

Let `F` be an unknown function — either a reduced-round block cipher `E_k` or a uniformly random permutation `π`.

Given input pairs with a fixed XOR difference:
```
(P, P ⊕ ΔP)  →  (C, C') = (F(P), F(P ⊕ ΔP))
```

**Goal:** Design an ML-based distinguisher that determines whether `F` is a real cipher or a random permutation by learning exploitable differential relations.

**Key insight:** Cipher outputs exhibit structured XOR differences due to the algebraic structure of the round function, while random permutations produce uniform noise. This structure is learnable.

---

## Cryptographic Setup

### Ciphers Implemented (8 total)

| Cipher | Block (bits) | Key (bits) | Full Rounds | Status |
|--------|-------------|-----------|-------------|--------|
| SIMON32/64 | 32 | 64 | 32 | PASS — test vector verified |
| GIFT-64 | 64 | 128 | 28 | PASS — non-zero output |
| GIFT-128 | 128 | 128 | 40 | PASS — non-zero output |
| SKINNY-64-64 | 64 | 64 | 32 | PASS — test vector verified |
| SKINNY-128-128 | 128 | 128 | 40 | PASS — non-zero output |
| CRAFT | 64 | 128 (K‖T) | 32 | PASS — non-zero output |
| Pyjamask-96 | 96 | 128 | 14 | PASS — non-zero output |
| PRESENT-80 | 64 | 80 | 31 | PASS — test vector verified |

### Avalanche Effect (all ciphers pass)

| Cipher | Avg bit flips | Ideal | Result |
|--------|--------------|-------|--------|
| SIMON32 | 15.44 | 16.0 | good |
| GIFT-64 | 31.69 | 32.0 | good |
| GIFT-128 | 66.62 | 64.0 | good |
| SKINNY-64 | 31.81 | 32.0 | good |
| SKINNY-128 | 65.75 | 64.0 | good |
| CRAFT | 32.19 | 32.0 | good |
| Pyjamask | 48.06 | 48.0 | good |
| PRESENT | 30.12 | 32.0 | good |

---

## Dataset Generation

Datasets are generated as balanced binary classification problems:

1. Sample random plaintext `P` using `os.urandom`
2. Compute paired plaintext `P' = P ⊕ ΔP` (fixed Gohr-style difference)
3. Encrypt both: `(C, C') = (E_k(P), E_k(P'))`
4. Label: `y = 1` (cipher) or `y = 0` (random permutation)
5. Apply chosen representation transform to `(C, C')`

**Fixed input differences (ΔP) per cipher:**

| Cipher | ΔP |
|--------|-----|
| simon32 | `0x40000000` |
| gift64/skinny64/craft/present | `0x0000000000000001` |
| gift128/skinny128 | `0x...0001` (128-bit) |
| pyjamask | `0x...0001` (96-bit) |

---

## Input Representations (7 total)

| Representation | Shape | Accuracy (MLP, r=4) | Notes |
|---------------|-------|---------------------|-------|
| **statistical** | `(59,)` | **0.9983** | 5-group feature vector (HW + nibble + XOR bits + autocorr) |
| joint | `(6×block_bits,)` | 0.9967 | `bits(P, P', C, C', ΔP, ΔC)` — white-box setting |
| **delta** | `(block_bits,)` | **0.9958** | `bits(C ⊕ C')` — best differential signal |
| raw | `(2×block_bits,)` | 0.9925 | Same as concat, explicit float32 dtype |
| bitslice | `(2×block_bits,)` | 0.9850 | Interleaved `[C_i, C'_i]` — CNN-friendly |
| concat | `(2×block_bits,)` | 0.9792 | `bits(C) ‖ bits(C')` — full information |
| word | `(3×n_words,)` | 0.9658 | 16-bit word-level normalised representation |

**Statistical representation features (59 features for simon32):**
- Group 1 (7): Global HW stats — HW(C), HW(C'), HW(ΔC), |ΔHW|, bit correlation, HW upper/lower half
- Group 2 (8): Per-nibble HW of ΔC
- Group 3 (32): Raw XOR bit vector `bits(C ⊕ C')`
- Group 4 (4): Per-byte HW of ΔC
- Group 5 (8): Autocorrelation of ΔC bits at lags 1–8

---

## Machine Learning Models (4 total)

### 1. MLP — Multi-Layer Perceptron
- **Architecture:** `Linear(1024) → LN → GELU × 5 + 2 skip connections → Linear(1) → Sigmoid`
- **Params:** 753,537 (delta rep) / 804,737 (concat rep)
- **Best rep:** delta
- **Best accuracy:** 1.0000 (r=4, delta, N=4000, 15 epochs)

### 2. CNN — Convolutional Neural Network
- **Architecture:** 3-channel input (C, C', XOR) → MultiScale stem (k=1,3,5) → 3 ResStages → global avg+max+std pooling → MLP head
- **Params:** 313,141
- **Best rep:** concat / bitslice
- **Best accuracy:** 0.9600 (r=4, concat, N=4000, 15 epochs)

### 3. Siamese Network
- **Architecture:** Shared encoder (2-layer) → concat embeddings + |diff| → classifier
- **Params:** 24,897
- **Best rep:** concat
- **Best accuracy:** 0.9975 (r=4, concat, N=4000, 15 epochs)

### 4. MINE — Mutual Information Neural Estimator
- **Architecture:** Augmented input `[C ‖ C' ‖ ΔC]` → `Linear(512) → LN → GELU × 4 + 2 skip connections → Linear(1) → Sigmoid`
- **Params:** 540,929
- **Best rep:** concat (augmented with XOR channel internally)
- **Best accuracy:** 0.9938 (r=4, concat, N=4000, 15 epochs)
- **Reference:** Belghazi et al., ICML 2018

---

## Experimental Results

### Experiment 1 — Accuracy vs Number of Rounds (simon32, concat)

| Rounds | MLP | CNN | Siamese | MINE |
|--------|-----|-----|---------|------|
| r=2 | 0.9400 | 1.0000 | 1.0000 | 1.0000 |
| r=3 | 0.5725 | 0.9675 | 0.9950 | 0.9988 |
| r=4 | 0.6900 | 0.8950 | 0.9700 | 0.9975 |
| r=5 | 0.5600 | 0.6050 | 0.7875 | 0.9375 |
| r=6 | 0.5425 | 0.5750 | 0.5425 | 0.6362 |
| r=7 | 0.5200 | 0.5375 | 0.5275 | 0.5400 |
| r=8 | 0.5150 | 0.4750 | 0.5300 | 0.5225 |
| r=9 | 0.5000 | 0.4750 | 0.5175 | 0.4950 |

**Max distinguishable round:** MLP r=5, CNN r=6, Siamese r=5, MINE r=6

### Multi-Cipher Comparison (MLP, delta rep)

| Cipher | Max Distinguishable Round | Accuracy |
|--------|--------------------------|---------|
| simon32 | r=8 | 0.5525 |
| gift64 | r=5 | 0.5725 |
| present | r=5 | 0.6225 |
| craft | r=3 | 0.9650 |
| pyjamask | r=2 | 0.6400 |
| skinny64 | r=6 | 0.5675 |
| gift128 | r=6 | 0.5950 |
| skinny128 | r=5 | 1.0000 |

### Experiment 2 — Accuracy vs Representation (MLP, simon32, r=4)

| Representation | Input Dim | Accuracy |
|---------------|-----------|---------|
| statistical | 59 | 0.9983 |
| delta | 32 | 0.9958 |
| joint | 192 | 0.9967 |
| raw | 64 | 0.9925 |
| bitslice | 64 | 0.9850 |
| concat | 64 | 0.9792 |
| word | 6 | 0.9658 |

### Experiment 2b — Best Accuracy per Cipher (MLP, delta, best round per cipher)

| Cipher | Round | Accuracy |
|--------|-------|---------|
| simon32 | r=5 | 0.9508 |
| gift64 | r=3 | 0.9933 |
| present | r=4 | 0.9200 |
| craft | r=3 | 0.9775 |
| pyjamask | r=1 | 1.0000 |
| skinny64 | r=5 | 1.0000 |
| gift128 | r=4 | 0.9900 |
| skinny128 | r=4 | 1.0000 |

### Experiment 3 — Accuracy vs Model (simon32, r=4, N=4000, 15 epochs)

| Model | Params | Rep | Accuracy |
|-------|--------|-----|---------|
| MLP | 753,537 | delta | 1.0000 |
| CNN | 313,141 | concat | 0.9600 |
| Siamese | 24,897 | concat | 0.9975 |
| MINE | 540,929 | concat | 0.9938 |

### Model Benchmark (simon32, r=4, concat, N=4000)

| Model | Best Accuracy | vs Baseline |
|-------|--------------|-------------|
| MLP | 1.0000 | +0.5000 |
| CNN | 0.9600 | +0.4600 |
| Siamese | 0.9975 | +0.4975 |
| MINE | 0.9938 | +0.4938 |
| Baseline (random) | 0.5000 | — |

### ML Distinguisher — Max Rounds per Cipher (MLP, delta)

| Cipher | Best Round | Best Accuracy | Max Dist. Round (multi-cipher scan) |
|--------|-----------|--------------|-------------------------------------|
| simon32 | r=5 | 0.9508 | r=8 (0.5525) |
| gift64 | r=3 | 0.9933 | r=5 (0.5725) |
| present | r=4 | 0.9200 | r=5 (0.6225) |
| craft | r=3 | 0.9775 | r=3 (0.9650) |
| pyjamask | r=1 | 1.0000 | r=2 (0.6400) |
| skinny64 | r=5 | 1.0000 | r=6 (0.5675) |
| gift128 | r=4 | 0.9900 | r=6 (0.5950) |
| skinny128 | r=4 | 1.0000 | r=5 (1.0000) |

---

## Project Structure

```
AC_Project/
├── neural_cryptanalysis/
│   ├── ciphers/
│   │   ├── base.py                  Abstract BlockCipher base class
│   │   ├── simon.py                 SIMON32/64 implementation
│   │   ├── gift64.py                GIFT-64 implementation
│   │   ├── gift128.py               GIFT-128 implementation
│   │   ├── skinny64.py              SKINNY-64-64 implementation
│   │   ├── skinny128.py             SKINNY-128-128 implementation
│   │   ├── craft.py                 CRAFT tweakable cipher
│   │   ├── pyjamask.py              Pyjamask-96 implementation
│   │   ├── present.py               PRESENT-80 implementation
│   │   ├── random_perm.py           Pseudo-random permutation baseline
│   │   ├── verify_all_ciphers.py    Full cipher verification + ML accuracy vs rounds
│   │   └── test_ciphers.py          Quick sanity tests for individual ciphers
│   ├── data/
│   │   ├── generator.py             Dataset generation, save, and load utilities
│   │   └── generate_all_datasets.py Bulk dataset generation script (all ciphers × reps × rounds)
│   ├── models/
│   │   ├── mlp.py                   Deep MLP with LayerNorm, GELU, skip connections
│   │   ├── cnn.py                   1D CNN with multi-scale stem and residual blocks
│   │   ├── siamese.py               Siamese twin-network distinguisher
│   │   └── mine.py                  MINE-based distinguisher with XOR augmentation
│   ├── representations/
│   │   ├── delta.py                 bits(C ⊕ C') — differential representation
│   │   ├── concat.py                bits(C) ‖ bits(C') — concatenation
│   │   ├── raw.py                   Same as concat, explicit float32
│   │   ├── bitslice.py              Interleaved bit channels — CNN-friendly
│   │   ├── word.py                  16-bit word-level normalised representation
│   │   ├── statistical.py           59-feature rich statistical vector
│   │   ├── joint.py                 Full plaintext+ciphertext white-box representation
│   │   └── utils.py                 int_to_bits, hamming_weight helpers
│   ├── experiments/
│   │   ├── run_all_experiments.py   Runs all 3 experiments + generates all plots
│   │   ├── benchmark_models.py      Detailed per-epoch benchmark of MLP/CNN/Siamese
│   │   ├── compare_representations.py  MLP accuracy across all 7 representations
│   │   ├── train_mlp.py             Standalone MLP training script
│   │   ├── train_cnn.py             Standalone CNN training script
│   │   ├── train_siamese.py         Standalone Siamese training script
│   │   └── train_mine.py            Standalone MINE training script
│   └── utils/
│       └── config.py                Cipher registry, DELTA_P, FULL_ROUNDS, get_cipher()
├── plots/
│   ├── acc_vs_rounds.png            Accuracy vs rounds for all 4 models (simon32)
│   ├── acc_vs_rounds_ciphers_a.png  Accuracy vs rounds (simon32/gift64/present/craft)
│   ├── acc_vs_rounds_ciphers_b.png  Accuracy vs rounds (pyjamask/skinny64/gift128/skinny128)
│   ├── acc_vs_rounds_all_ciphers.png All 8 ciphers accuracy vs rounds
│   ├── acc_vs_rounds_simon32.png    SIMON32 individual accuracy vs rounds
│   ├── acc_vs_rounds_gift64.png     GIFT-64 individual accuracy vs rounds
│   ├── acc_vs_rounds_present.png    PRESENT individual accuracy vs rounds
│   ├── acc_vs_rounds_craft.png      CRAFT individual accuracy vs rounds
│   ├── acc_vs_rounds_pyjamask.png   Pyjamask individual accuracy vs rounds
│   ├── acc_vs_rounds_skinny64.png   SKINNY-64 individual accuracy vs rounds
│   ├── acc_vs_rounds_gift128.png    GIFT-128 individual accuracy vs rounds
│   ├── acc_vs_rounds_skinny128.png  SKINNY-128 individual accuracy vs rounds
│   ├── acc_vs_representation.png    Accuracy vs representation type
│   ├── acc_vs_model.png             Accuracy vs model architecture
│   └── acc_vs_cipher.png            Best accuracy per cipher (all 8)
├── cipher_verification_output.txt   Cipher test vectors + avalanche + ML accuracy vs rounds
├── model_benchmark_output.txt       Per-epoch training log for MLP, CNN, Siamese
├── representation_comparison_output.txt  MLP accuracy across all 7 representations
├── experiments_output.txt           Full results for all 3 experiments
├── guide.txt                        Quick-start command reference
└── PROJECT_REPORT.md                This file
```

---

## How to Run

> All commands must be run from inside `AC_Project/`

### Step 1 — Install dependencies
```bash
pip install torch numpy tqdm matplotlib
```

### Step 2 — Verify all ciphers
```bash
python neural_cryptanalysis/ciphers/verify_all_ciphers.py
```
Runs test vectors, avalanche checks, differential outputs, and ML accuracy vs rounds for 5 ciphers. Saves to `cipher_verification_output.txt`.

### Step 3 — Generate a dataset
```bash
python -c "
from neural_cryptanalysis.data.generator import generate_dataset, save_dataset
from neural_cryptanalysis.utils.config   import get_cipher, DELTA_P

cipher = get_cipher('simon32')
X, y   = generate_dataset(cipher, rounds=4, n_samples=10000,
                           delta_p=DELTA_P['simon32'],
                           representation='delta')
save_dataset(X, y, 'simon_delta_r4')
print('Saved:', X.shape, y.shape)
"
```

**Available ciphers:** `simon32`, `gift64`, `gift128`, `skinny64`, `skinny128`, `craft`, `pyjamask`, `present`

**Available representations:** `delta`, `concat`, `raw`, `word`, `bitslice`, `statistical`, `joint`

### Step 4 — Train individual models
```bash
# MLP (works with any flat representation)
python -m neural_cryptanalysis.experiments.train_mlp \
    --dataset neural_cryptanalysis/data/datasets/simon_delta_r4 \
    --epochs 15 --batch 256 --lr 3e-4

# CNN (requires concat or raw representation)
python -m neural_cryptanalysis.experiments.train_cnn \
    --dataset neural_cryptanalysis/data/datasets/simon_concat_r4 \
    --epochs 10 --batch 128 --lr 1e-3

# Siamese (requires concat or raw representation)
python -m neural_cryptanalysis.experiments.train_siamese \
    --dataset neural_cryptanalysis/data/datasets/simon_concat_r4 \
    --epochs 10 --batch 128 --lr 1e-3

# MINE (requires concat or raw representation)
python -m neural_cryptanalysis.experiments.train_mine \
    --dataset neural_cryptanalysis/data/datasets/simon_concat_r4 \
    --epochs 15 --batch 256 --lr 3e-4
```

### Step 5 — Run full model benchmark
```bash
python -m neural_cryptanalysis.experiments.benchmark_models
```
Trains MLP, CNN, Siamese on the same dataset with per-epoch logging. Saves to `model_benchmark_output.txt`.

### Step 6 — Compare representations
```bash
python -m neural_cryptanalysis.experiments.compare_representations
```
Trains MLP on all 7 representations and compares accuracy. Saves to `representation_comparison_output.txt`.

### Step 7 — Run all experiments + generate plots
```bash
python -m neural_cryptanalysis.experiments.run_all_experiments
```
Runs all 3 experiments (accuracy vs rounds, vs representation, vs model) and saves 4 plots to `plots/`. Saves to `experiments_output.txt`.

### Step 8 — Generate all datasets in bulk (optional)
```bash
python neural_cryptanalysis/data/generate_all_datasets.py
# Optional flags:
# --cipher simon gift64       Only specific ciphers
# --repr delta concat         Only specific representations
# --n-samples 50000           Override sample count
# --no-skip                   Regenerate existing datasets
```

---

## Key Findings

1. **Statistical representation** achieves the highest accuracy (99.83%) — the 59-feature vector captures differential structure better than raw bits alone.
2. **Delta representation** is the most direct — encodes `C ⊕ C'` bits, achieving 99.58% at r=4 for simon32.
3. **MLP** achieves 100% accuracy at r=4 with delta rep — the simplest model wins with the right representation.
4. **Siamese network** is the most parameter-efficient model (25K params) with 99.75% accuracy.
5. **MINE** with XOR augmentation achieves 99.38% at r=4 — the explicit differential channel is critical.
6. **Cipher hardness** varies significantly: CRAFT becomes indistinguishable at r=4, while SIMON32 remains distinguishable up to r=8.
7. **pyjamask, skinny64, skinny128** are perfectly distinguishable (100%) at their best rounds.
8. All 8 ciphers pass test vector verification and avalanche effect checks.

---

## Output Files Reference

| File | Generated by | Description |
|------|-------------|-------------|
| `cipher_verification_output.txt` | `verify_all_ciphers.py` | Test vectors, avalanche, ML accuracy vs rounds |
| `model_benchmark_output.txt` | `benchmark_models.py` | Per-epoch training log for 3 models |
| `representation_comparison_output.txt` | `compare_representations.py` | MLP accuracy across 7 representations |
| `experiments_output.txt` | `run_all_experiments.py` | All 3 experiments + final summary |
| `plots/acc_vs_rounds.png` | `run_all_experiments.py` | Accuracy vs rounds (all 4 models, simon32) |
| `plots/acc_vs_rounds_ciphers_a.png` | `run_all_experiments.py` | Accuracy vs rounds (simon32/gift64/present/craft) |
| `plots/acc_vs_rounds_ciphers_b.png` | `run_all_experiments.py` | Accuracy vs rounds (pyjamask/skinny64/gift128/skinny128) |
| `plots/acc_vs_rounds_all_ciphers.png` | `run_all_experiments.py` | All 8 ciphers accuracy vs rounds |
| `plots/acc_vs_rounds_simon32.png` | `run_all_experiments.py` | SIMON32 individual accuracy vs rounds |
| `plots/acc_vs_rounds_gift64.png` | `run_all_experiments.py` | GIFT-64 individual accuracy vs rounds |
| `plots/acc_vs_rounds_present.png` | `run_all_experiments.py` | PRESENT individual accuracy vs rounds |
| `plots/acc_vs_rounds_craft.png` | `run_all_experiments.py` | CRAFT individual accuracy vs rounds |
| `plots/acc_vs_rounds_pyjamask.png` | `run_all_experiments.py` | Pyjamask individual accuracy vs rounds |
| `plots/acc_vs_rounds_skinny64.png` | `run_all_experiments.py` | SKINNY-64 individual accuracy vs rounds |
| `plots/acc_vs_rounds_gift128.png` | `run_all_experiments.py` | GIFT-128 individual accuracy vs rounds |
| `plots/acc_vs_rounds_skinny128.png` | `run_all_experiments.py` | SKINNY-128 individual accuracy vs rounds |
| `plots/acc_vs_representation.png` | `run_all_experiments.py` | Accuracy vs representation |
| `plots/acc_vs_model.png` | `run_all_experiments.py` | Accuracy vs model architecture |
| `plots/acc_vs_cipher.png` | `run_all_experiments.py` | Best accuracy per cipher (all 8) |
