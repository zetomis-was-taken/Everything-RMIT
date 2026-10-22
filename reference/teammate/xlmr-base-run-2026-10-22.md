# Teammate run output — XLM-R-base, 2026-10-22

Captured from the Kaggle web UI by the team. The notebook itself saves **no outputs**
(0 executed cells, empty Output tab), so this file is the only record of the numbers.

## Setup
- Backbone: `xlm-roberta-base` (Kaggle model `aisuko/xlm-roberta-base`), transformers 5.16.1
- Hardware: 2 x Tesla T4
- lr 2e-5, **epochs 5**, batch 16/GPU (effective 32), **maxlen 256**
- `USE_LANG_PREFIX=True` (`"[pt] text"`), class weights `[0.538, 1.075, 4.762]`
- Split: 10% holdout — 12,600 train / 1,400 val, seed 42
- 1,970 optimizer steps, 2,151.7 s wall → 0.916 steps/s (**1.09 s/step**)

## Per-epoch curve
| epoch | train loss | val loss | val macro-F1 |
|---|---|---|---|
| 1 | 0.2529 | 0.3212 | 0.9126 |
| 2 | 0.1955 | 0.2751 | 0.9186 |
| 3 | 0.1213 | 0.3128 | 0.9286 |
| **4** | 0.0475 | **0.2794** | **0.9493** |
| 5 | 0.0103 | 0.3064 | 0.9472 |

**Best epoch = 4; epoch 5 degrades both val loss and F1.** Train loss collapses to 0.010 by epoch 5 → overfitting.

## Best-model validation (n=1,400)
- macro-F1 (argmax) = **0.9493**
- per class: BENIGN 0.9691 (P 0.9800 / R 0.9585) · JAILBREAK 0.9517 (P 0.9278 / R 0.9770) · PROMPT_INJECTION 0.9271 (P 0.9468 / R 0.9082)
- per language: **en 0.8940 (n=139)** · fr 0.9753 (n=343) · it 0.9362 (n=288) · pt 0.9488 (n=630)
- confusion (55 errors total):
  - BENIGN: 832 correct, 31 → JAILBREAK, 5 → PI
  - JAILBREAK: 424 correct, 10 → BENIGN, 0 → PI
  - PROMPT_INJECTION: 89 correct, 7 → BENIGN, 2 → JAILBREAK
  - **BENIGN↔JAILBREAK = 41 of 55 errors (74.5%), asymmetry 31 vs 10 (3.1x toward JAILBREAK)**

## Threshold tuning (fitted on the same validation split → optimistic)
0.9493 → **0.9530** (+0.0037) with multipliers BENIGN 2.1 · JAILBREAK 2.25 · PROMPT_INJECTION 1.0

## Test prediction distribution
BENIGN 0.6133 · JAILBREAK 0.3217 · PROMPT_INJECTION 0.0650 (train prior: 0.62 / 0.31 / 0.07)

## Notes for us
- `submission.csv` was written but **not submitted** — the team is not on the leaderboard.
- `TrainingArguments` silently dropped `warmup_ratio` (transformers 5 removed it), so this run trained
  with **no warmup** — worth knowing when comparing against our `warmup_steps=126`.
- Effective learning signal: 5 epochs on 12.6k rows overfits by epoch 5; our config should prefer the
  best-checkpoint epoch rather than the last.
