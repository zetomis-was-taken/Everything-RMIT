"""Challenge 1 - multilingual prompt-attack classification (mmBERT-base, 3 classes).

Flow: pre-tokenize once -> class-weighted (or focal) cross-entropy -> per-fold training ->
save OUT-OF-FOLD and fold-averaged TEST probabilities (never labels) for later ensembling.

RUN="smoke" : 1 fold, 1 epoch, small subset - validates config and measures speed.
RUN="fold0" : full data, 1 fold - the real macro-F1 gate before committing to 5 folds.
RUN="full"  : the folds listed in FOLDS_TO_RUN (None = all) - shardable across two kernels,
              with TAG-suffixed artifacts merged afterwards by work/merge_shards.py.

transformers 5.16.1 API notes (verified against the installed wheel):
  * Trainer(processing_class=...)             - `tokenizer=` was removed in v5
  * TrainingArguments(eval_strategy=...)      - `evaluation_strategy=` was removed in v5
  * warmup_steps accepts a float ratio        - `warmup_ratio=` was removed in v5
  * compute_loss() takes num_items_in_batch   - override must match the base signature
  * no `datasets` dependency needed: a plain torch Dataset + a length-grouped
    sampler avoids both the HF-`datasets` dependency and the v5
    `train_sampling_strategy` API, while keeping dynamic padding (the real win:
    median prompt is 73 tokens, so fixed 384-token padding wastes ~5x compute).
"""

from __future__ import annotations

import json
import shutil
import time
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, Sampler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

RUN = "fold0"  # "smoke" | "fold0" | "full"

MODEL_DIR = "/kaggle/input/models/aisuko/mmbert-base/transformers/default/1"
DATA_DIR = "/kaggle/input/competitions/rmit-hackathon-2026"
OUT_DIR = "/kaggle/working"

LABELS = ["BENIGN", "JAILBREAK", "PROMPT_INJECTION"]
LABEL_TO_ID = {name: index for index, name in enumerate(LABELS)}
PROMPT_INJECTION_ID = LABEL_TO_ID["PROMPT_INJECTION"]

MAXLEN = 384  # measured coverage at the mmBERT tokenizer: 96.2% (256 would be 90.2%)
SEED = 42
N_FOLDS = 5
LEARNING_RATE = 2e-5
EPOCHS = 5  # raised from 3: another team's log improved monotonically 0.9236-0.9433-0.9517 at 3 epochs, and the teammate moved xlmr-base to 5
BATCH_SIZE = 16
# 6% of ~2,100 total steps. Passed as an int so the code is valid under both
# transformers 4.x (warmup_steps: int) and 5.x (int or float ratio).
WARMUP_STEPS = 126
USE_CLASS_WEIGHTS = True
# Explicit weights rather than "balanced": the OOF matrix shows BENIGN->JAILBREAK (502) is 1.8x the
# reverse, i.e. the JAILBREAK boost over-fires. WEIGHT_PRESET="decomposed" drops only the JAILBREAK
# boost and holds PROMPT_INJECTION, so the arm isolates one axis at a time.
WEIGHT_PRESET = "balanced"  # "balanced" | "decomposed" | "uniform"
DECOMPOSED_WEIGHTS = [0.54, 1.0, 4.76]  # BENIGN, JAILBREAK, PROMPT_INJECTION
LOSS_MODE = "weighted_ce"  # "weighted_ce" | "focal"
USE_LANG_PREFIX = False  # ablation: prepend "[pt] " etc. to the text
FOLDS_TO_RUN: list[int] | None = None  # None = all folds; [0, 1, 2] / [3, 4] to shard

SMOKE_TRAIN_ROWS = 1200
SMOKE_EVAL_ROWS = 400
BUCKET_MULTIPLIER = 50
FOCAL_GAMMA = 2.0


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def per_class_f1(reference: np.ndarray, prediction: np.ndarray, class_id: int) -> float:
    """F1 of a single class, written explicitly to avoid averaged-call ambiguity."""
    true_positive = float(np.sum((prediction == class_id) & (reference == class_id)))
    false_positive = float(np.sum((prediction == class_id) & (reference != class_id)))
    false_negative = float(np.sum((prediction != class_id) & (reference == class_id)))
    denominator = 2 * true_positive + false_positive + false_negative
    return 2 * true_positive / denominator if denominator > 0 else 0.0


def frame_texts(frame: pd.DataFrame) -> list[str]:
    """Raw texts, optionally prefixed with the language tag (ablation switch)."""
    texts = frame["text"].astype(str).tolist()
    if not USE_LANG_PREFIX:
        return texts
    languages = frame["language"].astype(str).tolist()
    return [f"[{language}] {text}" for language, text in zip(languages, texts, strict=True)]


class TextDataset(Dataset):
    """Pre-tokenized texts; returns model-ready tensors plus optional labels."""

    def __init__(
        self,
        encoded: list[dict[str, list[int]]],
        labels: list[int] | None = None,
    ) -> None:
        self.input_ids = [row["input_ids"] for row in encoded]
        self.attention_mask = [row["attention_mask"] for row in encoded]
        self.labels = labels
        self.lengths = [len(ids) for ids in self.input_ids]

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item: dict[str, Any] = {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
        }
        if self.labels is not None:
            item["labels"] = self.labels[index]
        return item


class SortishSampler(Sampler[int]):
    """Shuffle, cut into large buckets, sort each bucket by length.

    Gives near-sorted batches (so padding overhead collapses) while still mixing
    examples enough for stable gradients.
    """

    def __init__(self, lengths: list[int], batch_size: int, seed: int) -> None:
        self.lengths = lengths
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.lengths)

    def __iter__(self) -> Any:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        permutation = torch.randperm(len(self.lengths), generator=generator).tolist()
        bucket_size = max(self.batch_size * BUCKET_MULTIPLIER, self.batch_size)

        buckets: list[list[int]] = []
        for start in range(0, len(permutation), bucket_size):
            bucket = permutation[start : start + bucket_size]
            bucket.sort(key=lambda index: self.lengths[index])
            buckets.append(bucket)

        batch_order = torch.randperm(len(buckets), generator=generator).tolist()
        order: list[int] = []
        for bucket_index in batch_order:
            order.extend(buckets[bucket_index])
        return iter(order)


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Focal loss (gamma>0 down-weights easy examples); weight still applies per class."""
    log_probability = F.log_softmax(logits, dim=-1)
    target_log_probability = log_probability.gather(1, targets.unsqueeze(1)).squeeze(1)
    probability = target_log_probability.exp()
    modulation = (1.0 - probability).pow(FOCAL_GAMMA)
    per_example = -modulation * target_log_probability * weight[targets]
    return per_example.mean()


class WeightedTrainer(Trainer):
    """Cross-entropy (optionally focal) with class weights, which plain Trainer cannot express."""

    def __init__(
        self,
        *args: Any,
        class_weights: torch.Tensor,
        loss_mode: str = "weighted_ce",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.loss_mode = loss_mode
        self.sampler: SortishSampler | None = None

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        del num_items_in_batch  # weights already handle the imbalance
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weight = self.class_weights.to(outputs.logits.device, dtype=outputs.logits.dtype)
        if self.loss_mode == "focal":
            loss = focal_loss(outputs.logits, labels, weight)
        else:
            loss = nn.CrossEntropyLoss(weight=weight)(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss

    def _get_train_sampler(self, *args: Any, **kwargs: Any) -> Sampler[int] | None:
        dataset = args[0] if args else self.train_dataset
        lengths = getattr(dataset, "lengths", None)
        if not isinstance(lengths, list):
            return super()._get_train_sampler(*args, **kwargs)
        if self.sampler is None:
            self.sampler = SortishSampler(lengths, self.args.train_batch_size, self.args.seed)
            log(f"sortish sampler active: {len(lengths)} rows, batch={self.args.train_batch_size}, "
                f"len p50={int(np.percentile(lengths, 50))} p95={int(np.percentile(lengths, 95))}")
        return self.sampler


def build_folds(frame: pd.DataFrame) -> np.ndarray:
    folds = np.full(len(frame), -1, dtype=np.int64)
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (_, valid_index) in enumerate(splitter.split(frame, frame.label)):
        folds[valid_index] = fold
    return folds


def encode_texts(texts: list[str], tokenizer: PreTrainedTokenizerBase) -> list[dict[str, list[int]]]:
    encoded = tokenizer(texts, truncation=True, max_length=MAXLEN, return_attention_mask=True)
    batch = cast(dict[str, list[list[int]]], dict(encoded))
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    return [
        {"input_ids": input_ids[index], "attention_mask": attention_mask[index]}
        for index in range(len(input_ids))
    ]


def make_compute_metrics() -> Any:
    def compute_metrics(eval_prediction: EvalPrediction) -> dict[str, float]:
        logits = np.asarray(eval_prediction.predictions)
        reference = np.asarray(eval_prediction.label_ids)
        prediction = logits.argmax(axis=-1)
        result = {"macro_f1": float(f1_score(reference, prediction, average="macro"))}
        for name, class_id in LABEL_TO_ID.items():
            result[f"f1_{name.lower()}"] = per_class_f1(reference, prediction, class_id)
        return result

    return compute_metrics


def training_arguments(output_dir: str, epochs: int) -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
        learning_rate=LEARNING_RATE,
        num_train_epochs=epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=64,
        warmup_steps=WARMUP_STEPS,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        save_only_model=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        max_grad_norm=1.0,
        logging_steps=10,
        seed=SEED,
        data_seed=SEED,
        report_to="none",
        dataloader_num_workers=2,
    )


def run_fold(
    fold: int,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    test_dataset: TextDataset,
    tokenizer: PreTrainedTokenizerBase,
    epochs: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    log(f"fold {fold}: train={len(train_frame)} valid={len(valid_frame)} epochs={epochs}")

    train_labels = np.array([LABEL_TO_ID[name] for name in train_frame.label])
    if WEIGHT_PRESET == "balanced" and USE_CLASS_WEIGHTS:
        weights = compute_class_weight("balanced", classes=np.arange(len(LABELS)), y=train_labels)
    elif WEIGHT_PRESET == "decomposed":
        weights = np.array(DECOMPOSED_WEIGHTS, dtype=np.float64)
    else:
        weights = np.ones(len(LABELS), dtype=np.float64)
    log(f"fold {fold}: class weights {np.round(weights, 3).tolist()} loss={LOSS_MODE}")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id=LABEL_TO_ID,
        ignore_mismatched_sizes=True,
    )
    train_dataset = TextDataset(encode_texts(frame_texts(train_frame), tokenizer), train_labels.tolist())
    valid_dataset = TextDataset(
        encode_texts(frame_texts(valid_frame), tokenizer),
        [LABEL_TO_ID[name] for name in valid_frame.label],
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_arguments(f"{OUT_DIR}/out/fold{fold}", epochs),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
        processing_class=tokenizer,
        compute_metrics=make_compute_metrics(),
        class_weights=torch.tensor(weights, dtype=torch.float32),
        loss_mode=LOSS_MODE,
    )

    started = time.time()
    train_output = trainer.train()
    elapsed = time.time() - started
    log(f"fold {fold}: train_loss={train_output.training_loss:.4f} in {elapsed / 60:.1f} min")
    if torch.cuda.is_available():
        log(f"fold {fold}: peak_gpu_gb={torch.cuda.max_memory_allocated() / 1e9:.2f}")

    valid_logits = trainer.predict(valid_dataset).predictions
    test_logits = trainer.predict(test_dataset).predictions
    valid_probs = torch.softmax(torch.tensor(valid_logits, dtype=torch.float32), dim=-1).numpy()
    test_probs = torch.softmax(torch.tensor(test_logits, dtype=torch.float32), dim=-1).numpy()

    prediction = valid_probs.argmax(axis=-1)
    reference = np.array([LABEL_TO_ID[name] for name in valid_frame.label])
    report: dict[str, Any] = {
        "fold": fold,
        "seconds": round(elapsed, 1),
        "seconds_per_epoch": round(elapsed / epochs, 1),
        "class_weights": weights.round(4).tolist(),
        "macro_f1": float(f1_score(reference, prediction, average="macro")),
        "f1_by_class": [
            round(per_class_f1(reference, prediction, class_id), 4) for class_id in range(len(LABELS))
        ],
        "confusion": confusion_matrix(reference, prediction, labels=[0, 1, 2]).tolist(),
    }
    log(f"fold {fold}: macro_f1={report['macro_f1']:.4f} seconds_per_epoch={report['seconds_per_epoch']}")
    return valid_probs, test_probs, report


def artifact_tag(folds: list[int]) -> str:
    if len(folds) == N_FOLDS:
        return "all"
    return "f" + "_".join(str(fold) for fold in folds)


def main() -> None:
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    log(f"RUN={RUN} torch={torch.__version__} cuda={torch.cuda.is_available()} "
        f"gpus={torch.cuda.device_count()} device={device}")

    train_frame = pd.read_csv(f"{DATA_DIR}/train.csv")
    test_frame = pd.read_csv(f"{DATA_DIR}/test.csv")
    log(f"train={train_frame.shape} test={test_frame.shape}")
    log(f"label distribution: {train_frame.label.value_counts().to_dict()}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        log(f"pad_token was unset; using {tokenizer.pad_token!r}")
    log(f"tokenizer={type(tokenizer).__name__} vocab={tokenizer.vocab_size} "
        f"pad_id={tokenizer.pad_token_id} model_max_length={tokenizer.model_max_length}")

    train_frame = train_frame.assign(fold=build_folds(train_frame))
    try:
        train_frame[["id", "fold"]].to_csv(f"{OUT_DIR}/folds.csv", index=False)
    except OSError as error:
        log(f"could not write folds.csv: {error}")

    log("tokenizing test set once")
    test_dataset = TextDataset(encode_texts(frame_texts(test_frame), tokenizer))
    log(f"test tokens: p50={np.percentile(test_dataset.lengths, 50):.0f} "
        f"p95={np.percentile(test_dataset.lengths, 95):.0f} max={max(test_dataset.lengths)}")

    if RUN in ("smoke", "fold0"):
        folds = [0]
    else:
        folds = FOLDS_TO_RUN if FOLDS_TO_RUN is not None else list(range(N_FOLDS))
    epochs = 1 if RUN == "smoke" else EPOCHS
    tag = artifact_tag(folds)
    log(f"folds={folds} tag={tag}")

    oof_probs = np.zeros((len(train_frame), len(LABELS)), dtype=np.float32)
    test_probs = np.zeros((len(test_frame), len(LABELS)), dtype=np.float32)
    reports: list[dict[str, Any]] = []

    for fold in folds:
        is_valid = (train_frame.fold == fold).to_numpy()
        fold_train = cast(pd.DataFrame, train_frame[~is_valid])
        fold_valid = cast(pd.DataFrame, train_frame[is_valid])
        if RUN == "smoke":
            fold_train = cast(pd.DataFrame, fold_train.head(SMOKE_TRAIN_ROWS))
            fold_valid = cast(pd.DataFrame, fold_valid.head(SMOKE_EVAL_ROWS))

        valid_probs, fold_test_probs, report = run_fold(
            fold, fold_train, fold_valid, test_dataset, tokenizer, epochs
        )
        valid_rows = np.flatnonzero(is_valid)[: len(valid_probs)]
        oof_probs[valid_rows] = valid_probs
        test_probs += fold_test_probs / len(folds)
        reports.append(report)

    try:
        np.save(f"{OUT_DIR}/oof_probs_{tag}.npy", oof_probs)
        np.save(f"{OUT_DIR}/test_probs_{tag}.npy", test_probs)
    except OSError as error:
        log(f"could not write probability artifacts: {error}")

    covered = oof_probs.sum(axis=1) > 0
    summary: dict[str, Any] = {
        "run": RUN,
        "tag": tag,
        "folds_run": folds,
        "model_dir": MODEL_DIR,
        "maxlen": MAXLEN,
        "epochs": epochs,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "use_class_weights": USE_CLASS_WEIGHTS,
        "weight_preset": WEIGHT_PRESET,
        "loss_mode": LOSS_MODE,
        "use_lang_prefix": USE_LANG_PREFIX,
        "oof_rows_covered": int(covered.sum()),
        "folds": reports,
    }
    if covered.any():
        reference = np.array([LABEL_TO_ID[name] for name in train_frame.label])[covered]
        summary["oof_macro_f1"] = float(
            f1_score(reference, oof_probs[covered].argmax(axis=-1), average="macro")
        )
        log(f"OOF macro_f1 on {int(covered.sum())} rows: {summary['oof_macro_f1']:.4f}")

    try:
        with open(f"{OUT_DIR}/metrics_{tag}.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
    except OSError as error:
        log(f"could not write metrics json: {error}")

    # Checkpoints are ~1.2 GB per fold and are not needed: only probabilities matter,
    # and oversized output directories make `kaggle kernels output` crawl.
    try:
        shutil.rmtree(f"{OUT_DIR}/out", ignore_errors=True)
    except OSError as error:
        log(f"could not remove checkpoints: {error}")
    log(f"artifacts written: oof_probs_{tag}.npy, test_probs_{tag}.npy, metrics_{tag}.json")


if __name__ == "__main__":
    main()
