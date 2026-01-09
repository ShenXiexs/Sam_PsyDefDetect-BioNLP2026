# PsyDefDetect@BioNLP 2026 / 心理防御机制识别

## English

- **Task**: Given a multi-turn dialogue (`dialogue`) and a target utterance (`current_text`), predict one defense label (0–8) per DMRS (see `reference/Psychological Defense Mechanism Coding Handbook.pdf`). Train set: `input_data/train.json` (1,864 rows with labels). Test set: `input_data/test.json` (472 rows, unlabeled).
- **Labels**: 0 Neutral; 1 Action; 2 Major image-distorting; 3 Disavowal; 4 Minor image-distorting; 5 Neurotic; 6 Obsessional; 7 High-adaptive; 8 Needs more info. Train distribution: 7=51.93%, 0=15.88%, 6=9.23%, 1=5.79%, 3=5.31%, 4=4.51%, 2=3.27%, 5=2.58%, 8=1.50%. Avg context turns 13.2 (max 50); avg target words 19.4 (max 169).
- **Input formatting**: normalize spaces, keep casing; map speakers to `Seeker:`/`Supporter:`; wrap target in `<t>...</t>`; keep last N turns (e.g., 40 turns or ~448 tokens). If truncation loses core info, prepend a 1–2 sentence summary from a lightweight summarizer (BART/T5).
- **Data split**: 5-fold GroupKFold on `dialogue_id` to avoid leakage; optional dev split (10%) if single-run.
- **Models**:
  - Core classifier: `microsoft/deberta-v3-large`, max length 512, AdamW (lr 2e-5, warmup 10%, wd 0.01), batch 8–16, focal or weighted CE (weights ∝ 1/√freq), R-Drop (λ≈0.5), fp16.
  - Long context: `allenai/longformer-base-4096` (or deberta-long), max length 1024, lr 1e-5, batch 4.
  - NLI variant: premise = dialogue+target, hypothesis = label definition; start from `microsoft/deberta-v3-large-mnli`, train 2-class entailment, then normalize scores across 9 labels.
- **Imbalance & regularization**: class weights + minority oversampling; context dropout (drop 1 prior turn) for robustness; label smoothing 0.05; light EDA only for rare classes.
- **Semi-supervised**: pseudo-label high-confidence test samples (p≥0.9, focus on minority) and fine-tune 1–2 epochs with low weight (~0.3).
- **Ensemble**: softmax-average probs from (a) core cls, (b) long context, (c) NLI; optional weight by CV scores; temperature-scale on dev.
- **Outputs**: `submission.jsonl` lines `{"id": "...", "label": int}` aligned to test order. Keep experiment log + model card noting clinical sensitivity.

### Quickstart (commands)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U transformers datasets accelerate evaluate scikit-learn sentencepiece

# Core classifier (fold 0 of 5) — GPU/large RAM; on Mac MPS, prefer the base model below
python scripts/train.py \
  --model microsoft/deberta-v3-large --max-length 512 --max-turns 40 \
  --lr 2e-5 --batch 8 --num-epochs 6 --loss focal --r-drop 0.5 \
  --group-field dialogue_id --num-folds 5 --fold-index 0 \
  --output-dir outputs/deberta-v3-f0

# Lighter run for Mac MPS / constrained VRAM
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 \
python scripts/train.py \
  --model microsoft/deberta-v3-base --max-length 384 --max-turns 32 \
  --lr 2e-5 --batch 1 --grad-accum 16 --num-epochs 6 --loss focal --r-drop 0.5 \
  --group-field dialogue_id --num-folds 5 --fold-index 0 \
  --output-dir outputs/deberta-v3-base-f0

# Long-context
python scripts/train.py \
  --model allenai/longformer-base-4096 --max-length 1024 --max-turns 40 \
  --lr 1e-5 --batch 4 --num-epochs 4 \
  --group-field dialogue_id --num-folds 5 --fold-index 0 \
  --output-dir outputs/longformer-f0

# NLI-style
python scripts/train_nli.py \
  --model microsoft/deberta-v3-large-mnli --max-length 512 --max-turns 40 \
  --batch 16 --num-epochs 4 \
  --group-field dialogue_id --num-folds 5 --fold-index 0 \
  --output-dir outputs/nli-f0

# Ensemble inference (adjust checkpoints/weights as needed)
python scripts/predict.py \
  --test-file input_data/test.json \
  --checkpoints outputs/deberta-v3-f0 outputs/longformer-f0 outputs/nli-f0 \
  --weights 1 1 1 --temperature 0.9 --out submission.jsonl
```

## 中文

- **任务**：给定多轮对话 (`dialogue`) 与目标发言 (`current_text`)，预测 9 类防御标签（0–8，详见手册 PDF）。训练集 `input_data/train.json`（1864 条，有标签），测试集 `input_data/test.json`（472 条，无标签）。
- **标签**：0 中性/无防御；1 行为型；2 严重形象扭曲；3 否认/推脱；4 轻微形象扭曲；5 神经质；6 强迫型；7 高度适应；8 信息不足。训练分布：7=51.93%，0=15.88%，6=9.23%，1=5.79%，3=5.31%，4=4.51%，2=3.27%，5=2.58%，8=1.50%；平均上下文 13.2 轮，目标约 19 词。
- **输入构造**：清理空格、保留大小写；说话人统一为 `Seeker:`/`Supporter:`；目标用 `<t>...</t>` 包裹；保留末尾 N 轮（如 40 轮或 ~448 token），必要时先摘要再拼接。
- **切分**：按 `dialogue_id` 做 5 折 GroupKFold，避免同对话泄漏；或 9:1 随机划分做单次开发集。
- **模型**：
  - 主分类：`microsoft/deberta-v3-large`，512 长度，AdamW(lr 2e-5, warmup 10%, wd 0.01)，batch 8–16，focal/加权 CE（1/√频次），R-Drop(0.5)，fp16。
  - 长上下文：`allenai/longformer-base-4096`（或 deberta-long），长度 1024，lr 1e-5，batch 4。
  - NLI 变体：前提=对话+目标，假设=标签定义；基于 MNLI 预训练模型训练 2 分类，9 个假设归一化得标签概率。
- **不平衡/正则**：类别权重+少数类过采样；上下文 dropout；标签平滑 0.05；少数类轻量 EDA。
- **半监督**：对测试高置信 (p≥0.9) 样本伪标签，低权重再训 1–2 轮，优先补齐少样本。
- **集成**：核心 + 长上下文 + NLI softmax 加权平均，验证集做温度标定。
- **输出**：`submission.jsonl`，每行 `{"id": "...", "label": int}`，顺序与测试集一致；附实验日志和模型说明（含伦理/临床敏感性提示）。

### 快速开始（命令同上，参数可按需调整）

> Mac MPS 内存紧张时可用轻量命令：`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python scripts/train.py --model microsoft/deberta-v3-base --max-length 384 --max-turns 32 --batch 1 --grad-accum 16 ...`
