import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional
import sys

# Ensure project root on sys.path when running as a script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from scripts.data_utils import LABEL_DEFS, compute_class_weights, format_dialogue, load_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train classifier for PsyDefDetect.")
    parser.add_argument("--train-file", default="input_data/train.json")
    parser.add_argument("--val-file", default=None, help="Optional explicit validation file.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="microsoft/deberta-v3-large")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loss", choices=["ce", "focal"], default="ce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--r-drop", type=float, default=0.0, help="R-Drop KL weight, 0 to disable.")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--context-dropout", type=float, default=0.0)
    parser.add_argument("--group-field", default=None, help="Use GroupKFold on this field (e.g., dialogue_id).")
    parser.add_argument("--num-folds", type=int, default=1, help="If >1, use GroupKFold and pick --fold-index.")
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Used if no val-file and num-folds=1.")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument("--evaluation-steps", type=int, default=None)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--metric-for-best", default="macro_f1")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience (eval steps).")
    parser.add_argument("--oversample-minority", action="store_true")
    return parser.parse_args()


def build_dataset(
    data: List[Dict[str, Any]],
    tokenizer,
    max_length: int,
    max_turns: Optional[int],
    context_dropout: float,
) -> Dataset:
    def _map(examples):
        texts = []
        for dial, current in zip(examples["dialogue"], examples["current_text"]):
            texts.append(
                format_dialogue(
                    dialogue=dial, current_text=current, max_turns=max_turns, target_tag=True, context_dropout=context_dropout
                )
            )
        tokenized = tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=max_length,
        )
        return tokenized

    ds = Dataset.from_list(data)
    remove_cols = [c for c in ds.column_names if c != "label"]
    return ds.map(_map, batched=True, remove_columns=remove_cols)


class WeightedTrainer(Trainer):
    def __init__(
        self,
        *args,
        class_weights: Optional[torch.Tensor] = None,
        focal_gamma: float = 2.0,
        r_drop: float = 0.0,
        label_smoothing: float = 0.0,
        train_sampler_override=None,
        loss_name: str = "ce",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.focal_gamma = focal_gamma
        self.r_drop = r_drop
        self.label_smoothing = label_smoothing
        self.train_sampler_override = train_sampler_override
        self.loss_name = loss_name

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None and self.class_weights.device != logits.device:
            self.class_weights = self.class_weights.to(logits.device)

        loss = self._compute_primary_loss(logits, labels)
        if self.r_drop > 0:
            outputs2 = model(**inputs)
            logits2 = outputs2.logits
            loss2 = self._compute_primary_loss(logits2, labels)
            kl = F.kl_div(F.log_softmax(logits, dim=-1), F.softmax(logits2, dim=-1), reduction="batchmean")
            kl += F.kl_div(F.log_softmax(logits2, dim=-1), F.softmax(logits, dim=-1), reduction="batchmean")
            loss = 0.5 * (loss + loss2) + 0.5 * self.r_drop * kl

        return (loss, outputs) if return_outputs else loss

    def _compute_primary_loss(self, logits, labels):
        if self.loss_func == "focal":
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            ce = F.nll_loss(log_probs, labels, weight=self.class_weights, reduction="none")
            pt = probs.gather(1, labels.unsqueeze(-1)).squeeze(-1)
            focal = (1 - pt) ** self.focal_gamma * ce
            return focal.mean()
        return F.cross_entropy(
            logits,
            labels,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )

    @property
    def loss_func(self) -> str:
        return self.loss_name

    def get_train_dataloader(self):
        if self.train_sampler_override is None:
            return super().get_train_dataloader()
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=self.train_sampler_override,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    macro_f1 = f1_score(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    return {"macro_f1": macro_f1, "accuracy": acc}


def maybe_build_sampler(labels: List[int]) -> Optional[WeightedRandomSampler]:
    counts = Counter(labels)
    max_count = max(counts.values())
    weights = [max_count / counts[l] for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def make_split(
    data: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> (List[Dict], List[Dict]):
    if args.val_file:
        train_data = data
        val_data = load_json(args.val_file)
        return train_data, val_data

    if args.num_folds > 1 and args.group_field:
        groups = [d[args.group_field] for d in data]
        splitter = GroupKFold(n_splits=args.num_folds)
        all_splits = list(splitter.split(data, groups=groups))
        if args.fold_index >= len(all_splits):
            raise ValueError("fold_index out of range for given num_folds")
        train_idx, val_idx = all_splits[args.fold_index]
        return [data[i] for i in train_idx], [data[i] for i in val_idx]

    train_data, val_data = train_test_split(
        data,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=[d["label"] for d in data],
    )
    return train_data, val_data


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_json(args.train_file)
    train_data, val_data = make_split(raw, args)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    train_labels = [d["label"] for d in train_data]
    class_weights = compute_class_weights(train_labels)
    class_weights_t = torch.tensor(class_weights, dtype=torch.float)

    train_ds = build_dataset(
        train_data,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_turns=args.max_turns,
        context_dropout=args.context_dropout,
    )
    val_ds = build_dataset(
        val_data,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_turns=args.max_turns,
        context_dropout=0.0,
    )

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=len(LABEL_DEFS))
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.num_epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="steps" if args.evaluation_steps else "epoch",
        eval_steps=args.evaluation_steps,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        save_strategy="steps" if args.evaluation_steps else "epoch",
        load_best_model_at_end=True,
        metric_for_best_model=args.metric_for_best,
        greater_is_better=True,
        fp16=args.fp16,
        report_to="none",
    )

    sampler = None
    if args.oversample_minority:
        sampler = maybe_build_sampler(train_labels)

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights_t,
        focal_gamma=args.focal_gamma,
        r_drop=args.r_drop,
        label_smoothing=args.label_smoothing,
        train_sampler_override=sampler,
        loss_name=args.loss,
    )

    if args.patience:
        from transformers.trainer_utils import IntervalStrategy
        from transformers.trainer_callback import EarlyStoppingCallback

        if training_args.evaluation_strategy == IntervalStrategy.NO:
            raise ValueError("Set evaluation-steps or use epoch evaluation to enable early stopping.")
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=args.patience))

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Save label mapping and metadata for predict.py
    metadata = {
        "label_defs": LABEL_DEFS,
        "model_type": "cls",
        "train_args": vars(args),
    }
    Path(output_dir, "metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
