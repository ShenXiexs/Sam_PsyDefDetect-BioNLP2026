import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
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
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from scripts.data_utils import LABEL_DEFS, format_pair_input, load_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NLI-style training for PsyDefDetect.")
    parser.add_argument("--train-file", default="input_data/train.json")
    parser.add_argument("--val-file", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="microsoft/deberta-v3-large-mnli")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pos-weight-scale", type=float, default=1.0, help="Multiply positive class weight by this factor.")
    parser.add_argument("--r-drop", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--group-field", default=None)
    parser.add_argument("--num-folds", type=int, default=1)
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--metric-for-best", default="macro_f1")
    parser.add_argument("--evaluation-steps", type=int, default=None)
    return parser.parse_args()


def make_split(
    data: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Tuple[List[Dict], List[Dict]]:
    if args.val_file:
        return data, load_json(args.val_file)

    if args.num_folds > 1 and args.group_field:
        groups = [d[args.group_field] for d in data]
        splitter = GroupKFold(n_splits=args.num_folds)
        splits = list(splitter.split(data, groups=groups))
        if args.fold_index >= len(splits):
            raise ValueError("fold_index out of range")
        train_idx, val_idx = splits[args.fold_index]
        return [data[i] for i in train_idx], [data[i] for i in val_idx]

    train_data, val_data = train_test_split(
        data,
        test_size=args.val_ratio,
        random_state=args.seed,
        stratify=[d["label"] for d in data],
    )
    return train_data, val_data


def expand_pairs(data: List[Dict], max_turns: Optional[int]) -> List[Dict]:
    pairs = []
    for ex_idx, d in enumerate(data):
        for label_id, label_def in LABEL_DEFS.items():
            premise, hyp = format_pair_input(
                dialogue=d["dialogue"],
                current_text=d["current_text"],
                hypothesis=label_def,
                max_turns=max_turns,
            )
            pairs.append(
                {
                    "premise": premise,
                    "hypothesis": hyp,
                    "label": 1 if label_id == d["label"] else 0,
                }
            )
    return pairs


def tokenize_pairs(ds: Dataset, tokenizer, max_length: int) -> Dataset:
    def _map(examples):
        tokenized = tokenizer(
            examples["premise"],
            examples["hypothesis"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        return tokenized

    remove_cols = [c for c in ds.column_names if c != "label"]
    return ds.map(_map, batched=True, remove_columns=remove_cols)


class NLITrainer(Trainer):
    def __init__(
        self,
        *args,
        class_weights: Optional[torch.Tensor] = None,
        r_drop: float = 0.0,
        label_smoothing: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.r_drop = r_drop
        self.label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None and self.class_weights.device != logits.device:
            self.class_weights = self.class_weights.to(logits.device)

        loss = F.cross_entropy(
            logits,
            labels,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )
        if self.r_drop > 0:
            outputs2 = model(**inputs)
            logits2 = outputs2.logits
            loss2 = F.cross_entropy(
                logits2,
                labels,
                weight=self.class_weights,
                label_smoothing=self.label_smoothing,
            )
            kl = F.kl_div(F.log_softmax(logits, dim=-1), F.softmax(logits2, dim=-1), reduction="batchmean")
            kl += F.kl_div(F.log_softmax(logits2, dim=-1), F.softmax(logits, dim=-1), reduction="batchmean")
            loss = 0.5 * (loss + loss2) + 0.5 * self.r_drop * kl

        return (loss, outputs) if return_outputs else loss


def compute_pair_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    macro_f1 = f1_score(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    return {"macro_f1": macro_f1, "accuracy": acc}


def aggregate_eval(
    model,
    tokenizer,
    val_data: List[Dict[str, Any]],
    max_turns: Optional[int],
    max_length: int,
) -> Dict[str, float]:
    device = next(model.parameters()).device
    model.eval()
    all_probs = []
    all_labels = []

    features = []
    for idx, d in enumerate(val_data):
        for label_id, label_def in LABEL_DEFS.items():
            premise, hyp = format_pair_input(d["dialogue"], d["current_text"], label_def, max_turns=max_turns)
            features.append(
                {
                    "example_id": idx,
                    "label_id": label_id,
                    "premise": premise,
                    "hypothesis": hyp,
                    "gold": d["label"],
                }
            )

    def _collate(batch):
        enc = tokenizer(
            [b["premise"] for b in batch],
            [b["hypothesis"] for b in batch],
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        enc["example_id"] = torch.tensor([b["example_id"] for b in batch], dtype=torch.long)
        enc["label_id"] = torch.tensor([b["label_id"] for b in batch], dtype=torch.long)
        enc["gold"] = torch.tensor([b["gold"] for b in batch], dtype=torch.long)
        return enc

    loader = DataLoader(features, batch_size=32, shuffle=False, collate_fn=_collate)
    entailment_scores = {}
    with torch.no_grad():
        for batch in loader:
            example_ids = batch.pop("example_id")
            label_ids = batch.pop("label_id")
            gold = batch.pop("gold")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            scores = logits.softmax(dim=-1)[:, 1]  # entailment prob
            for e_id, l_id, g, s in zip(example_ids, label_ids, gold, scores.cpu()):
                entailment_scores.setdefault(int(e_id), {})[int(l_id)] = float(s)
                if len(all_labels) <= e_id:
                    all_labels.append(int(g))

    for ex_id in range(len(all_labels)):
        label_scores = entailment_scores[ex_id]
        scores = np.array([label_scores[i] for i in sorted(label_scores.keys())])
        probs = scores / scores.sum()
        all_probs.append(probs)

    preds = [int(np.argmax(p)) for p in all_probs]
    macro_f1 = f1_score(all_labels, preds, average="macro")
    acc = accuracy_score(all_labels, preds)
    return {"macro_f1": macro_f1, "accuracy": acc}


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_json(args.train_file)
    train_data, val_data = make_split(raw, args)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    train_pairs = expand_pairs(train_data, max_turns=args.max_turns)
    val_pairs = expand_pairs(val_data, max_turns=args.max_turns)

    # class weights: positive class is rare (1 per 9)
    labels = [p["label"] for p in train_pairs]
    counts = Counter(labels)
    neg, pos = counts.get(0, 1), counts.get(1, 1)
    pos_weight = (neg / pos) * args.pos_weight_scale
    class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float)

    train_ds = tokenize_pairs(Dataset.from_list(train_pairs), tokenizer, args.max_length)
    val_ds = tokenize_pairs(Dataset.from_list(val_pairs), tokenizer, args.max_length)

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
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

    trainer = NLITrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_pair_metrics,
        class_weights=class_weights,
        r_drop=args.r_drop,
        label_smoothing=args.label_smoothing,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    agg_metrics = aggregate_eval(
        model=trainer.model,
        tokenizer=tokenizer,
        val_data=val_data,
        max_turns=args.max_turns,
        max_length=args.max_length,
    )
    Path(output_dir, "aggregate_eval.json").write_text(json.dumps(agg_metrics, indent=2))

    metadata = {
        "label_defs": LABEL_DEFS,
        "model_type": "nli",
        "train_args": vars(args),
    }
    Path(output_dir, "metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
