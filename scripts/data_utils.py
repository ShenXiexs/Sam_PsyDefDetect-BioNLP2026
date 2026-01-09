import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

LABEL_DEFS: Dict[int, str] = {
    0: "Neutral / No Defense: functional utterances that keep the conversation going without emotional defense.",
    1: "Action Defenses: discharging distress through actions (passive aggression, help-rejecting complaining, acting out).",
    2: "Major Image-Distorting: extreme all-good/all-bad distortion of self or others (splitting, projective identification).",
    3: "Disavowal: denial, rationalization, projection, or autistic fantasy to avoid painful reality.",
    4: "Minor Image-Distorting: milder distortion of self/other (devaluation, idealization, omnipotence).",
    5: "Neurotic: repression, dissociation, reaction formation, displacement; feelings emerge indirectly.",
    6: "Obsessional: intellectualization, isolation of affect, undoing; excessive control of feelings through logic/symbolic acts.",
    7: "High-Adaptive: mature coping (affiliation, altruism, anticipation, humor, self-assertion, self-observation, sublimation, suppression).",
    8: "Needs More Information / Unclear: insufficient or ambiguous content to classify.",
}


def load_json(path: str) -> List[Dict]:
    return json.loads(Path(path).read_text())


def truncate_dialogue(dialogue: Sequence[Dict], max_turns: Optional[int]) -> List[Dict]:
    if max_turns is None or len(dialogue) <= max_turns:
        return list(dialogue)
    return list(dialogue[-max_turns:])


def format_dialogue(
    dialogue: Sequence[Dict],
    current_text: str,
    max_turns: Optional[int] = None,
    target_tag: bool = True,
    context_dropout: float = 0.0,
) -> str:
    """
    Build a single string prompt from a multi-turn dialogue and target utterance.
    context_dropout randomly drops one prior turn to improve robustness.
    """
    turns = truncate_dialogue(dialogue, max_turns=max_turns)
    if context_dropout > 0 and len(turns) > 2 and random.random() < context_dropout:
        drop_idx = random.randrange(0, len(turns) - 1)  # never drop last turn
        turns = turns[:drop_idx] + turns[drop_idx + 1 :]

    parts = []
    for t in turns:
        speaker = t.get("speaker", "speaker")
        speaker = {"seeker": "Seeker", "supporter": "Supporter"}.get(speaker.lower(), speaker)
        parts.append(f"{speaker}: {t['text']}")

    target = current_text
    if target_tag:
        target = f"<t>{current_text}</t>"
    parts.append(f"Target: {target}")

    instruction = (
        "Task: classify the psychological defense level (0-8) of the target utterance given the prior dialogue context. "
        "Use context to infer the defense function rather than surface emotion."
    )
    return instruction + "\n" + "\n".join(parts)


def format_pair_input(
    dialogue: Sequence[Dict],
    current_text: str,
    hypothesis: str,
    max_turns: Optional[int] = None,
) -> Tuple[str, str]:
    """
    Build (premise, hypothesis) pair for NLI-style training/inference.
    Premise = dialogue context + target utterance, Hypothesis = label definition.
    """
    premise = format_dialogue(dialogue, current_text, max_turns=max_turns, target_tag=True, context_dropout=0.0)
    return premise, hypothesis


def compute_class_weights(labels: Sequence[int], method: str = "sqrt") -> np.ndarray:
    counts = Counter(labels)
    weights = []
    for i in range(len(LABEL_DEFS)):
        c = counts.get(i, 1)
        if method == "sqrt":
            weights.append(1.0 / np.sqrt(c))
        else:
            weights.append(1.0 / c)
    weights = np.array(weights, dtype=np.float32)
    weights = weights * (len(LABEL_DEFS) / weights.sum())  # normalize scale
    return weights


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
