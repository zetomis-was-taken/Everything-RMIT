# Challenge 1 — status brief

**Competition:** [rmit-hackathon-2026](https://www.kaggle.com/competitions/rmit-hackathon-2026) — Security x Generative AI x Low-Resource Languages
**Task:** classify a prompt as `BENIGN` / `JAILBREAK` / `PROMPT_INJECTION`, in pt / fr / it / en
**Metric:** macro-F1 (each class weighted equally, so the 7% `PROMPT_INJECTION` class is 1/3 of the score)
**Deadline:** 2026-10-22 11:00 UTC
**Team:** `RMIT_SGS_Ae-do-toc_C1` (confirmed) · kernel `tkhoikk/rmit-sgs-ae-do-toc-c1`

At the time of writing: **0 submissions spent**, 1 smoke test run on GPU, TF-IDF baseline measured locally.

---

## 1. Where we stand

| item | status |
|---|---|
| Data downloaded & profiled | done |
| CPU baseline (TF-IDF + LinearSVC) | done — **0.9195** macro-F1 (5-fold OOF; a 15% hold-out had said 0.9209) |
| GPU pipeline validated end-to-end | done — smoke test, no errors, no NaN |
| Submission budget used | **0 of 6** (today's allowance) |
| Real transformer score | **not yet measured** — that's the next honest number |

## 2. What the labels actually mean

Derived from the data, not from the description (see `work/explore.py`):

| label | median length | does what | lexical signature |
|---|---|---|---|
| `BENIGN` | 115 chars (p90 1,112; max 18,550) | two populations: casual asks **and** long content-production templates ("write an SEO article in [TARGETLANGUAGE], use H1/H2/H3, 5 FAQs") | `targetlanguage`, `lingua di`, `h1 h2`, `faq` |
| `JAILBREAK` | 565 chars / 86 words | *argues for* disallowed content via roleplay & hypotheticals | `dear model`, `contradictory`, `subtly`, `self-harm` |
| `PROMPT_INJECTION` | 99 chars | *targets the instruction channel*: probes its own prompt ("what was your initial directive?") or overrides it ("ignore the above and say…") | `pwned`/`pwn`, `above sentence`, `respond only` |

`BENIGN` = legitimate task templates. `JAILBREAK` = content-policy violation.
`PROMPT_INJECTION` = instruction-channel attack. The last two differ *pragmatically*, not topically.

## 3. Evidence collected (all measured, nothing assumed)

**Class balance** — BENIGN 62.0% · JAILBREAK 31.0% · PROMPT_INJECTION 7.0%, identical ratio in every language.
Balanced-loss weights on 14k rows: `[0.54, 1.08, 4.76]`.

**Token budget** (XLM-R tokenizer, train set) — this is why maxlen matters:

| max_len | 128 | 192 | 256 | 320 | **384** | 512 |
|---|---|---|---|---|---|---|
| coverage | 63.6% | 80.4% | 88.9% | 93.3% | **95.8%** | 97.9% |

p50 = 73 tokens, p90 = 268, p98 = 522, max = 4,132. The BTC sample uses 256, which silently truncates
11% of prompts — and the long ones are the long jailbreaks. We use **384**.

**Where the cheap model fails** (LinearSVC, **5-fold OOF over all 14,000 rows** — every row was predicted by a model that never saw it):

```text
pred              BENIGN  JAILBREAK  PROMPT_INJECTION
true
BENIGN              8149        502                29
JAILBREAK            284       4054                 2
PROMPT_INJECTION     132         15               833
```

**81.5% of all errors (786/964) are `BENIGN↔JAILBREAK`.** `PROMPT_INJECTION` is already **0.9035** F1 — it is the
*easy* class lexically despite being rare. Per-class: BENIGN 0.9451 · JAILBREAK 0.9099 · PROMPT_INJECTION 0.9035.

Two things the matrix says that a macro-F1 number hides:

1. **The B↔J error is asymmetric: 502 BENIGN→JAILBREAK vs 284 JAILBREAK→BENIGN (1.8×).** That fingerprint is
   consistent with class weighting pushing decisions off BENIGN, not with pure semantic ambiguity — so milder
   class weights (or per-class multipliers) are a concrete lever, not a hunch.
2. **PROMPT_INJECTION barely confuses with JAILBREAK at all** (2 and 15 cases); its only real confusion is with
   BENIGN (132). PI errors total 178/964 = 18.5% of the error mass, and they are a *recall* problem.
   This is why a high-precision PI-vs-rest router is the wrong architecture here: it cannot touch the B↔J
   boundary by construction, and its precision bias would drop exactly the 132 PI rows we already miss.

**Per-language (5-fold OOF, same model):** pt **0.8963** (n=6300) · it 0.9329 (n=2800) · en 0.9294 (n=1400) ·
fr 0.9468 (n=3500). **Portuguese is the weakest and it is also the longest-text language** — truncation risk and
weakness coincide. An earlier 15% hold-out had reported `en 0.880` as worst; that was hold-out noise on 213 rows
and did **not** replicate at 5 folds. Lesson applied: single hold-out numbers on this dataset are not trustworthy.

**Corpus structure:** *not* parallel machine translation. Nearest neighbour is same-language 385/400
times; a cross-language neighbour matches the label only 65.7% (≈ the 62% base rate). No translation-pair
shortcut exists, and translation augmentation has no free lunch.

**Language column is trustworthy**: 98.0% agreement with `langdetect` on a 600-row sample.

**Tokenizer fertility per language** (measured on train, `chars/token` — higher is more efficient):

| tokenizer | en | fr | it | pt | cov@256 | cov@384 |
|---|---|---|---|---|---|---|
| mmBERT-base (256k Gemma2) | 4.61 | 4.21 | 4.18 | 4.59 | 90.2% | **96.2%** |
| XLM-R base (250k sentencepiece) | 4.10 | 3.82 | 4.09 | 4.37 | 88.9% | 95.8% |

Romance languages cost ~10% more tokens than English under mmBERT (fr/it), but Portuguese is
indistinguishable from English — and under XLM-R Portuguese is the *most* efficient. mmBERT's 256k vocab is
~12% more token-efficient overall than XLM-R's. Median token counts differ a lot by language: pt `p50 = 136`,
en `p50 = 45` — Portuguese is the truncation-risk language, not French/Italian.

**GPU smoke test** (mmBERT-base, 1,200 rows, 1 epoch, maxlen 384):

| metric | value |
|---|---|
| throughput | 0.933 steps/s (1.07 s/step), effective batch 32 → both T4s used |
| peak GPU | 10.7 GB / 15 GB |
| wall clock incl. model download | ~5 min |
| macro-F1 | 0.441 — meaningless as accuracy (1 epoch, 1/10th data), but proves the pipeline |

Projection for a real fold: 11,200 ÷ 32 = 350 steps/epoch × 3 epochs × 1.07 s ≈ **19 min/fold**, ≈ 95 min for 5 folds.

## 4. Competition reality (pulled from the API, 03:40 UTC)

| our score would be | rank of 57 |
|---|---|
| 0.9209 (our TF-IDF) | **~51** |
| 0.9500 | ~30 |
| 0.9600 | ~27 |
| 0.9700 | **~12** |
| 0.9750 | ~6 |
| 0.9800 | ~3 |
| 0.9836 | 1 (current leader) |

Median 0.9538 · 25th percentile 0.9322. **The TF-IDF baseline is bottom-quartile here** — a transformer
is the entry ticket, not an optimisation.

**The "6 submissions" in the brief is 6 *per day*, not 6 total** —
`competition_get_submission_limits` returns `{'_num_allowed_now': 6, '_limited_by_total': False}`, and
one leaderboard team has 8 submissions. Practically we still have only 6 before today's deadline.

## 5. Files

| path | what it is |
|---|---|
| `C1-multilingual-prompt-attack.md` | the original brief (Vietnamese) |
| `work/data/` | train.csv, test.csv, sample_submission.csv |
| `work/explore.py` | the data-understanding script behind §2–§3 |
| `work/tfidf_baseline.py` | the 0.9209 CPU baseline |
| `kernels/rmit-sgs-ae-do-toc-c1/train.py` | **the training script** (read this) |
| `kernels/rmit-sgs-ae-do-toc-c1/kernel-metadata.json` | name, attachments, GPU flags |
| `work/make_submission.py` | merges fold shards, honest tuning check, writes submission.csv |
| `reference/teammate/` | the teammate's notebook, for comparison |
| `reference/*publicleaderboard*.csv` | leaderboard snapshot |

## 6. Reading the training script

`kernels/rmit-sgs-ae-do-toc-c1/train.py`, top to bottom:

- **`RUN`** — `"smoke"` (1 fold, 1 epoch, subset) · `"fold0"` (full data, 1 fold = the real gate) · `"full"` (5 folds).
- **`LABELS`** — order *is* the class index. Never `pd.factorize`; alphabetical order silently permutes classes.
- **`per_class_f1`** — written by hand so there's no ambiguity about which average was used.
- **`TextDataset`** — pre-tokenized tensors, no HF `datasets` dependency.
- **`SortishSampler`** — shuffle → bucket → sort each bucket by length. Median prompt is 73 tokens, so
  fixed 384-token padding would burn ~5× the compute. This is where the free GPU time comes from.
- **`WeightedTrainer.compute_loss`** — class-weighted cross-entropy; signature matches transformers 5.x
  (`num_items_in_batch`).
- **`build_folds`** — 5-fold stratified, seed 42. One frozen split reused by every run, so model
  comparisons are apples-to-apples.
- **`run_fold`** — fresh model per fold, train, then save **probabilities** (not labels) for validation and test.
- **`main`** — folds → OOF matrix + fold-averaged test probabilities → `oof_probs.npy`, `test_probs.npy`,
  `metrics.json`; checkpoints deleted before exit (each is ~1.2 GB).

transformers 5.16.1 traps that were verified against the installed wheel and are already handled:
`processing_class=` (not `tokenizer=`), `eval_strategy=` (not `evaluation_strategy=`), `warmup_ratio` **removed**,
`compute_loss` gained `num_items_in_batch`, `save_only_model=True`.

## 7. Open decisions

1. **Team name — resolved:** `RMIT_SGS_Ae-do-toc_C1`, confirmed by the team. It is genuinely unreachable via
   the API (a team only materialises on the leaderboard after its first submission, and neither `tkhoikk` nor
   the teammate's username appears in any of the 57 member lists), so it had to come from the team. The
   teammate's own notebook is titled `RMIT_SGS_TEAM_C1`, which no longer matches the team name — worth flagging.
2. **Which backbone gets the 5-fold run** — mmBERT-base (most modern multilingual base) vs wolf-defender
   (same architecture, initialised from a prompt-injection guard, so it may transfer its prior to pt/fr/it).
   Costs ~20 min to answer with real numbers.
3. **Class-multiplier tuning** — worth +0.01–0.04 on imbalanced macro-F1 *if* fitted honestly
   (tune on 4 folds, score on the 5th). Fitted on the reported split it is just noise-fitting.
   `work/make_submission.py` implements the honest version and falls back to raw argmax if the gain
   does not survive held-out scoring.
4. **The 5-fold run will be sharded** across two kernels (folds 0–2, folds 3–4) for ~50 min wall-clock
   instead of ~95. Each shard writes `oof_probs_<tag>.npy` / `test_probs_<tag>.npy` / `metrics_<tag>.json`;
   `work/make_submission.py` asserts the shards cover disjoint folds before merging.

## 8. Risks

- `is_frozen: True` on the competition. Teams are still scoring (a rank-3 submission landed at 03:38), so it
  does not block submissions — but confirm on our first submit.
- DataParallel across 2 T4s gives ~1.3–1.5×, not 2× (single process). Peak memory already 10.7 GB, so
  `xlm-roberta-large`/`rembert` will need gradient checkpointing.
- Long `BENIGN` tails (max 18,550 chars) are still truncated at 384; the tail sits on the hardest boundary.

---

## 9. Audit of received advice (claims vs. measured evidence)

A long technical brief was shared with the team. Principles are sound; several load-bearing specifics do not
survive contact with this dataset.

| claim | verdict | evidence |
|---|---|---|
| macro-F1 ⇒ `PROMPT_INJECTION` (7%) is 1/3 of the score | **right** | it is the metric |
| argmax under-predicts rare classes; tune thresholds | **right in principle** | we use balanced weights `[0.54, 1.08, 4.76]` |
| LLM/LoRA is a trap in an 8h window | **right** | 2×T4, no bf16, 2.5–7B |
| safety-tuning priors are English-centric | **right** | DeBERTa card: "solely prompt injections in English" |
| "only 6 submissions" | **wrong** | API: `_num_allowed_now: 6, _limited_by_total: False`; a team has 8 |
| "256 tokens is likely sufficient" | **wrong** | measured coverage 90.2% (mmBERT) / 88.9% (XLM-R) |
| "random split leaks language patterns; stratify by language" | **wrong premise** | class ratio identical (62/31/7) in all 4 languages; not parallel translations (nearest neighbour same-language 385/400; cross-language label agreement 65.7% ≈ base rate) |
| "a 10% holdout is enough to decide" | **too noisy** | 1,400 rows ⇒ ±0.02 macro-F1, larger than the gaps that separate the top of the board |
| "Romance languages tokenize much worse than English" | **directionally true, minor** | ~10% for fr/it; pt ≈ en (see §3 table) |
| "RemBERT is a sleeper for low-resource" | **misapplied** | pt/fr/it are high-resource for general NLP; RemBERT is the slowest encoder here (575M, 32 layers, batch 8) |
| "deberta-pi: multilingual F1 0.800, 33% false-negative" | **unverified** | not on the card; card reports English-only F1 95.49%, recall 99.74%, precision 91.59% |
| "language-stratified batching / per-language heads" | **low priority** | the implied failure mode does not appear: at 5-fold OOF, **fr 0.947** (best) and **en 0.929** (mid) — not English failing on Romance; **pt 0.896** is the weak spot and it is a *length/truncation* problem, not a language-coverage one |
| "threshold tuning is a big free win" | **right, but needs discipline** | tuning 3 multipliers on the same rows you report can manufacture +0.03 of noise; hence the 4-fold-tune/held-out-score check |

**The brief's real blind spot:** it targets `PROMPT_INJECTION` recall, but 79% of errors (106/134) are
`BENIGN↔JAILBREAK`, and `PROMPT_INJECTION` is already 0.896 F1 with a TF-IDF model because "ignore the above
and say X" is lexically distinctive. The B↔J boundary is semantic — which is precisely why a multilingual
encoder is mandatory rather than optional.

**Adopted from the brief:** one focal-loss ablation arm (cheap, testable, may help the B↔J margin).
**Discarded:** 10% holdout, maxlen 256, language-stratified splitting, RemBERT, un-nested threshold tuning.

**Language-prefix ablation (measured on the cheap model, 5-fold OOF):** raw text 0.9195 → with `[pt]`/`[fr]`
prefix 0.9226, i.e. **+0.0031**. That is inside fold-to-fold noise (fold spread 0.9136–0.9254, ~±0.005), so
the honest verdict is *inconclusive but free to test* — it gets one arm on the transformer, not a rewrite.

---

## 10. To-do

Status 04:35 UTC, deadline 11:00 UTC (~6h20m), **0 of 6 daily submissions used**.

### Blocked on the two gates finishing
- [ ] Gate logs: per-epoch macro-F1 curve, `seconds_per_epoch`, peak GPU, and the `sortish sampler active` line.
- [ ] Set the epoch count from that curve, not from an assumption of 3 or 5.
- [ ] Compare 384 vs 512 on overall macro-F1, **pt per-language F1**, PI recall and B→J count.
      Judge 512 on pt + PI, not macro. XLM-R caps at 514 positions; mmBERT's RoPE has no wall.
- [ ] Re-derive real cost per 5-fold from the measured `seconds_per_epoch` before queueing production.

### Interventions (single fold, judged on confusion counts, not per-language F1)
- [ ] Decomposed weights `[0.54, 1.0, 4.76]`: accept only if B→J falls **and** PI→BENIGN (132) does not rise.
- [ ] Focal loss γ=2 with **reduced** weights (never compound with 4.76).
- [ ] XLM-R-base **fold-0 A/B** (not a 5-fold); if competitive, treat it as a blend member, not a replacement.

### Rejection rule (replaces "no language drops >0.02")
Per-language macro-F1 at fold-0 size is too noisy for a 0.02 threshold. Bootstrapped 95% CI width:
pt 0.040 · fr 0.054 · it 0.070 · **en 0.095** (en n=280, std 0.025). So at fold level judge the
**B→J and PI→B counts** (~100 and ~26 per fold, Poisson ±10%); make per-language claims only on full
5-fold OOF (en n=1400 → ±0.011).

### Paired-comparison rule (supersedes the σ arithmetic above)
Comparisons run on the **same rows** have correlated errors, so the noise on a *delta* is not the noise on
either estimate, and a Poisson model on a marginal count is the wrong yardstick. Use `work/compare_pair.py`,
which resamples rows and recomputes **both** F1s per resample, and reports exact McNemar on discordant pairs.

384 vs 512 on fold 0 (2,800 rows), corrected:

| subset | n | 384 | 512 | delta | paired 95% CI | McNemar | p |
|---|---|---|---|---|---|---|---|
| ALL | 2800 | 0.9612 | 0.9669 | +0.0057 | [−0.0037, +0.0151] | 26 / 42 | 0.068 |
| pt | 1265 | 0.9503 | 0.9565 | +0.0062 | [−0.0048, +0.0168] | 16 / 28 | 0.096 |
| fr | 713 | 0.9617 | 0.9689 | +0.0073 | [−0.0143, +0.0310] | 4 / 6 | 0.754 |
| it | 566 | 0.9640 | 0.9906 | +0.0266 | [+0.0071, +0.0517] | 1 / 7 | 0.070 |
| en | 256 | 0.9969 | 0.9598 | −0.0370 | [−0.0806, −0.0038] | 5 / 1 | 0.219 |
| **B\|J** | 2604 | 0.6431 | 0.6475 | +0.0044 | [+0.0005, +0.0087] | 21 / 39 | **0.027** |

Reading: no per-language delta is individually supported (CIs cross zero for pt/fr; it and en exclude zero on
CI but have 8 and 6 discordant rows, so McNemar disagrees). The one result that holds up is the **B↔J
sub-problem** — restricted to the 2,604 rows whose true label is BENIGN or JAILBREAK, 512 wins with p=0.027.
So 512 is *free and helps the boundary that holds 81% of the errors*, which is a stronger justification than
"nearly free". Arms must be compared to **512's counts**, via McNemar on B↔J discordance, and reported with
the full 3×3 matrix (a weight change can trade one cell for another and still pass a two-cell check).

### Production
- [ ] Sharded 5-fold on the winning config (folds 0–2 / 3–4, two kernels).
- [ ] Merge shards; honest 4-fold multiplier tuning, kept only if it survives held-out scoring.
- [ ] If XLM-R is competitive: 5-fold it and blend on OOF, accepted only if OOF improves with no language regression.

### External data points (use instead of spending lanes)
- [x] **XLM-R-base A/B — satisfied by the teammate run** (`reference/teammate/xlmr-base-run-2026-10-22.md`):
      maxlen 256, 5 epochs, 10% holdout → best epoch 4, **val macro-F1 0.9493** (0.9530 with multipliers
      fitted on that same split). Do not spend a lane re-running it; a 5-fold XLM-R is only worth it as a
      **blend member**.
- [x] `thanhsu263` XLM-R-base log: val 0.9517 at 3 epochs, pt 0.9329 lowest.
- [x] Host demo `aisuko/rmit-sgs-demo-c1`: xlm-roberta-base on **200 rows**, 1 epoch — a smoke demo, not a baseline.

### What those data points change
- **Epochs:** the teammate curve peaks at **4** (0.9493) and degrades at 5 (0.9472, val loss 0.2794 → 0.3064,
  train loss 0.010). Prefer best-checkpoint selection; treat 5 epochs as the ceiling, not the target.
- **Throughput reconciled:** their 1.09 s/step at maxlen 256 vs our 1.07 s/step at 384 → the length-grouped
  sampler makes us roughly 1.6x more token-efficient. My earlier "1.6x slower than expected" concern was
  a bad comparison against a fixed-padding run.
- **Per-language ranking is unstable across runs:** ours (TF-IDF 5-fold) and `thanhsu263` both put **pt** last
  (0.896 / 0.933), while the teammate run puts **en** last (0.894, n=139). Three runs, two different "worst"
  languages → per-language verdicts must wait for 5-fold OOF. pt is the most reliable one per fold (n=1260, std 0.0096).
- **The B↔J story holds across model families:** 74.5% of the transformer's errors (41/55) and 81.5% of
  TF-IDF's (786/964), with the same direction of asymmetry (3.1x toward JAILBREAK here). The weight arm stays aimed correctly.
- **Their run trained with no warmup** — transformers 5 silently dropped `warmup_ratio`. Our `warmup_steps=126` is intact.
- **Cost re-derived:** at 1.07 s/step, fold 0 is 350 steps/epoch → 5 epochs ≈ 31 min/fold, 4 epochs ≈ 25 min/fold.
  Sharded 5-fold (3+2 folds across two lanes) is therefore **~95 min at 5 epochs, ~75 min at 4** — not the 50 min I
  projected from 3 epochs.

### Submission
- [ ] Build `submission.csv` with the assertions intact (6000 rows, label set, id order, no NaN).
- [ ] Confirm the real team name appears on the leaderboard after the first upload (validates naming).
- [ ] Team approval before every upload; 6 per day.

### Coordination
- [ ] Teammate notebook is titled `RMIT_SGS_AnhEmDoToc_C1` while the confirmed team name is `Ae-do-toc` — needs their confirmation.
- [ ] Their notebook is `COMPLETE` but has **no executed cells and no outputs**; ask them to Save Version with outputs
      or share their val numbers, so their XLM-R run counts as evidence instead of being duplicated.
- [ ] Regenerate the logbook and push after each material change (`work/make_logbook.py`).

### Explicitly out of scope
PI-vs-rest routing (leaves all 786 B↔J errors untouched), per-language heads/thresholds (identical class ratios
per language), LLM/LoRA fine-tuning (no bf16, 7h window), RemBERT, DeBERTa-PI as primary, 10% holdout, maxlen 256.
