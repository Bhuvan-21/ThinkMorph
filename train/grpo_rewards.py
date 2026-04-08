"""
Reward functions for GRPO training on ThinkMorph interleaved reasoning.

Each reward function scores a model completion against a ground-truth answer.
Rewards operate only on the final <answer>…</answer> text extracted from
the model's output.
"""

import re
from abc import ABC, abstractmethod
from typing import Optional


def extract_answer(text: str) -> Optional[str]:
    """Extract content inside the last <answer>…</answer> tags."""
    matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    """Check for: The answer is XX (or is: XX).  If so, extract XX."""
    matches = re.findall(r"answer is[:\s]+(.*?)(?:\s|\.|$)", text, re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return None


def normalize_text(text: str) -> str:
    """Lowercase, strip whitespace and punctuation for fuzzy comparison."""
    text = text.lower().strip()
    text = re.sub(r"[()]", "", text)  # remove parentheses
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class RewardFunction(ABC):
    @abstractmethod
    def __call__(self, prediction: str, ground_truth: str) -> float:
        """Score a single completion. Returns a float reward."""
        ...


class ExactMatchReward(RewardFunction):
    """1.0 if normalized predicted answer == normalized ground truth, else 0.0."""

    def __call__(self, prediction: str, ground_truth: str) -> float:
        # gt_answer = extract_answer(ground_truth)
        gt_answer = ground_truth
        pred_answer = extract_answer(prediction)
        # print(f"GT answer: '{gt_answer}' | Pred answer: '{pred_answer}'")
        if gt_answer is None or pred_answer is None:
            return 0.0
        return 1.0 if normalize_text(pred_answer) == normalize_text(gt_answer) else 0.0


class SoftMatchReward(RewardFunction):
    """Reward based on longest common subsequence ratio for partial credit."""

    def __call__(self, prediction: str, ground_truth: str) -> float:
        # gt_answer = extract_answer(ground_truth)
        gt_answer = ground_truth
        pred_answer = extract_answer(prediction)
        if gt_answer is None or pred_answer is None:
            return 0.0
        a = normalize_text(pred_answer)
        b = normalize_text(gt_answer)
        if not a or not b:
            return 0.0
        lcs_len = self._lcs_length(a, b)
        return (2.0 * lcs_len) / (len(a) + len(b))

    @staticmethod
    def _lcs_length(a: str, b: str) -> int:
        m, n = len(a), len(b)
        prev = [0] * (n + 1)
        for i in range(1, m + 1):
            curr = [0] * (n + 1)
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(prev[j], curr[j - 1])
            prev = curr
        return prev[n]


class FormatReward(RewardFunction):
    """Small bonus for having a well-formed <answer> tag, regardless of content."""

    def __init__(self, correct_bonus: float = 1.0, format_bonus: float = 0.1):
        self.correct_bonus = correct_bonus
        self.format_bonus = format_bonus

    def __call__(self, prediction: str, ground_truth: str) -> float:
        # gt_answer = extract_answer(ground_truth)
        gt_answer = ground_truth
        pred_answer = extract_answer(prediction)
        if gt_answer is None or pred_answer is None:
            return 0.0
        reward = self.format_bonus
        if normalize_text(pred_answer) == normalize_text(gt_answer):
            reward += self.correct_bonus
        return reward


REWARD_REGISTRY = {
    "exact_match": ExactMatchReward,
    "soft_match": SoftMatchReward,
    "format": FormatReward,
}


def get_reward_fn(name: str, **kwargs) -> RewardFunction:
    """Instantiate a reward function by name."""
    if name not in REWARD_REGISTRY:
        raise ValueError(f"Unknown reward function '{name}'. Available: {list(REWARD_REGISTRY.keys())}")
    return REWARD_REGISTRY[name](**kwargs)
