import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import sys

# Ensure project root on sys.path when running as a script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from scripts.data_utils import LABEL_DEFS, format_dialogue, format_pair_input, load_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference and ensembling for PsyDefDetect.")
    parser.add_argument("--test-file", default="input_data/test.json")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="List of checkpoint paths.")
    parser.add_argument("--weights", nargs="*", type=float, help="Optional weights for each checkpoint.")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--out", default="submission.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="Override device, e.g., cuda:0")
    return parser.parse_args()


def load_metadata(path: Path) -> Dict[str, Any]:
    meta_path = path / "metadata.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return {"model_type": "cls", "label_defs": LABEL_DEFS}


def build_cls_dataset(data: List[Dict], tokenizer, max_length: int, max_turns: int) -> Dataset:
    def _map(examples):
        texts = []
        for dial, current in zip(examples["dialogue"], examples["current_text"]):
            texts.append(format_dialogue(dialogue=dial, current_text=current, max_turns=max_turns, target_tag=True))
        tokenized = tokenizer(texts, padding=False, truncation=True, max_length=max_length)
        return tokenized

    ds = Dataset.from_list(data)
    remove_cols = [c for c in ds.column_names if c != "label_placeholder"]
    ds = ds.add_column("label_placeholder", [0] * len(ds))
    return ds.map(_map, batched=True, remove_columns=remove_cols)


def predict_cls(
    model,
    tokenizer,
    data: List[Dict],
    max_length: int,
    max_turns: int,
    batch_size: int,
    device,
    temperature: float,
) -> np.ndarray:
    ds = build_cls_dataset(data, tokenizer, max_length=max_length, max_turns=max_turns)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collator)
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if k != "label_placeholder"}
            logits = model(**batch).logits / temperature
            prob = torch.softmax(logits, dim=-1)
            probs.append(prob.cpu().numpy())
    return np.concatenate(probs, axis=0)


def predict_nli(
    model,
    tokenizer,
    data: List[Dict],
    max_length: int,
    max_turns: int,
    batch_size: int,
    device,
    temperature: float,
) -> np.ndarray:
    features = []
    for ex_idx, d in enumerate(data):
        for label_id, label_def in LABEL_DEFS.items():
            premise, hyp = format_pair_input(
                dialogue=d["dialogue"],
                current_text=d["current_text"],
                hypothesis=label_def,
                max_turns=max_turns,
            )
            features.append(
                {
                    "example_id": ex_idx,
                    "label_id": label_id,
                    "premise": premise,
                    "hypothesis": hyp,
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
        return enc

    loader = DataLoader(features, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    entailment_scores = {}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            example_ids = batch.pop("example_id")
            label_ids = batch.pop("label_id")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits / temperature
            scores = torch.softmax(logits, dim=-1)[:, 1]  # entailment probability
            for e_id, l_id, s in zip(example_ids, label_ids, scores.cpu()):
                entailment_scores.setdefault(int(e_id), {})[int(l_id)] = float(s)

    probs = []
    for ex_id in range(len(data)):
        label_scores = entailment_scores.get(ex_id, {})
        ordered = np.array([label_scores[i] for i in sorted(label_scores.keys())])
        if ordered.sum() == 0:
            ordered = np.ones_like(ordered) / len(ordered)
        else:
            ordered = ordered / ordered.sum()
        probs.append(ordered)
    return np.stack(probs, axis=0)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_data = load_json(args.test_file)

    ckpt_paths = [Path(p) for p in args.checkpoints]
    weights = args.weights if args.weights else [1.0] * len(ckpt_paths)
    if len(weights) != len(ckpt_paths):
        raise ValueError("Number of weights must match number of checkpoints.")

    all_probs = []
    for path, w in zip(ckpt_paths, weights):
        meta = load_metadata(path)
        model_type = meta.get("model_type", "cls")
        tokenizer = AutoTokenizer.from_pretrained(str(path), use_fast=False)
        model = AutoModelForSequenceClassification.from_pretrained(str(path))
        model.to(device)

        if model_type == "nli":
            probs = predict_nli(
                model=model,
                tokenizer=tokenizer,
                data=test_data,
                max_length=args.max_length,
                max_turns=args.max_turns,
                batch_size=args.batch,
                device=device,
                temperature=args.temperature,
            )
        else:
            probs = predict_cls(
                model=model,
                tokenizer=tokenizer,
                data=test_data,
                max_length=args.max_length,
                max_turns=args.max_turns,
                batch_size=args.batch,
                device=device,
                temperature=args.temperature,
            )
        all_probs.append((w, probs))

    weight_sum = sum(w for w, _ in all_probs)
    ensemble = sum(w * p for w, p in all_probs) / weight_sum
    preds = np.argmax(ensemble, axis=-1)

    out_path = Path(args.out)
    with out_path.open("w") as f:
        for item, pred in zip(test_data, preds):
            f.write(json.dumps({"id": item["id"], "label": int(pred)}) + "\n")
    print(f"Wrote predictions to {out_path}")


if __name__ == "__main__":
    main()
