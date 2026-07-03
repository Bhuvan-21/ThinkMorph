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


def parse_number(text: str) -> Optional[float]:
    """Pull the first signed decimal out of `text`, ignoring thousands commas,
    currency symbols, and trailing units (e.g. '$1,234.5 billion' -> 1234.5).
    Returns None if no numeric token is present."""
    if text is None:
        return None
    m = re.search(r"[-+]?\d*\.?\d+", text.replace(",", ""))
    return float(m.group()) if m else None


def lcs_similarity(a: str, b: str) -> float:
    """Longest-common-subsequence overlap in [0, 1]: 2·LCS / (|a| + |b|)."""
    if not a or not b:
        return 0.0
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
    return (2.0 * prev[n]) / (m + n)


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
        gt_answer = ground_truth
        pred_answer = extract_answer(prediction)
        if gt_answer is None or pred_answer is None:
            return 0.0
        return lcs_similarity(normalize_text(pred_answer), normalize_text(gt_answer))


class GradedReward(RewardFunction):
    """Exact match scores 1.0; otherwise partial credit in [partial_low, partial_high].

    Partial credit is graded by closeness so that "almost right" answers — which
    binary exact-match throws away — still carry a learning signal, while never
    tying a true exact match:
      - both sides numeric: by relative error, clamped to 0.0 beyond `num_rel_tol`.
        rel_err→0 approaches `partial_high`; rel_err→num_rel_tol gives `partial_low`.
      - otherwise (text):   by LCS overlap, clamped to 0.0 below `text_sim_floor`.
        overlap→1 approaches `partial_high`; overlap→floor gives `partial_low`.
    """

    def __init__(
        self,
        partial_low: float = 0.5,
        partial_high: float = 0.8,
        num_rel_tol: float = 0.5,
        text_sim_floor: float = 0.5,
    ):
        self.partial_low = partial_low
        self.partial_high = partial_high
        self.num_rel_tol = num_rel_tol
        self.text_sim_floor = text_sim_floor

    def _scale(self, frac: float) -> float:
        """Map a closeness fraction in [0, 1] onto [partial_low, partial_high]."""
        return self.partial_low + (self.partial_high - self.partial_low) * frac

    def __call__(self, prediction: str, ground_truth: str) -> float:
        gt_answer = ground_truth
        pred_answer = extract_answer(prediction)
        if gt_answer is None or pred_answer is None:
            return 0.0

        # Exact (normalized) match always wins.
        if normalize_text(pred_answer) == normalize_text(gt_answer):
            return 1.0

        # Numeric distance when both sides are numbers.
        gt_num, pred_num = parse_number(gt_answer), parse_number(pred_answer)
        if gt_num is not None and pred_num is not None:
            denom = max(abs(gt_num), 1e-8)
            rel_err = abs(pred_num - gt_num) / denom
            if rel_err >= self.num_rel_tol:
                return 0.0
            return self._scale(1.0 - rel_err / self.num_rel_tol)

        # Text overlap otherwise.
        sim = lcs_similarity(normalize_text(pred_answer), normalize_text(gt_answer))
        if sim <= self.text_sim_floor:
            return 0.0
        return self._scale((sim - self.text_sim_floor) / (1.0 - self.text_sim_floor))


class GradedSharpReward(RewardFunction):
    """Sharpened variant of GradedReward. Exact match → 1.0; otherwise a *thin*
    partial in [partial_low, partial_high] so that exact strongly dominates and
    GRPO is pushed toward exact answers rather than coasting on near-misses
    (the partial-credit "dead-end" where a confident near-miss scores high, never
    reaches exact, and — being consistent across the group — yields no gradient).

      - numeric: numerically-equal → 1.0 (handles e.g. 34.60 vs 34.6, 1,234 vs 1234);
        within a TIGHT num_rel_tol → graded by closeness into [low, high]; else 0.0.
      - text: near-binary — only near-identical (LCS ≥ text_sim_floor) earns a small
        partial; anything genuinely different → 0.0.
      - MCQ (single letters): falls through to text and scores 0.0 unless exact,
        i.e. effectively binary, same as before.
    """

    def __init__(
        self,
        partial_low: float = 0.1,
        partial_high: float = 0.5,
        num_rel_tol: float = 0.05,
        text_sim_floor: float = 0.85,
        num_exact_eps: float = 1e-4,
    ):
        self.partial_low = partial_low
        self.partial_high = partial_high
        self.num_rel_tol = num_rel_tol
        self.text_sim_floor = text_sim_floor
        self.num_exact_eps = num_exact_eps

    def _scale(self, frac: float) -> float:
        """Map a closeness fraction in [0, 1] onto [partial_low, partial_high]."""
        return self.partial_low + (self.partial_high - self.partial_low) * frac

    def __call__(self, prediction: str, ground_truth: str) -> float:
        gt_answer = ground_truth
        pred_answer = extract_answer(prediction)
        if gt_answer is None or pred_answer is None:
            return 0.0

        # Exact (normalized) match always wins.
        if normalize_text(pred_answer) == normalize_text(gt_answer):
            return 1.0

        # Numeric path: treat numerically-equal as exact, then a tight near-miss band.
        gt_num, pred_num = parse_number(gt_answer), parse_number(pred_answer)
        if gt_num is not None and pred_num is not None:
            rel_err = abs(pred_num - gt_num) / max(abs(gt_num), 1e-8)
            if rel_err <= self.num_exact_eps:
                return 1.0
            if rel_err < self.num_rel_tol:
                return self._scale(1.0 - rel_err / self.num_rel_tol)
            return 0.0

        # Text path: near-binary — only near-identical strings get a small partial.
        sim = lcs_similarity(normalize_text(pred_answer), normalize_text(gt_answer))
        if sim < self.text_sim_floor:
            return 0.0
        return self._scale((sim - self.text_sim_floor) / (1.0 - self.text_sim_floor))


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
    "graded": GradedReward,
    "graded_sharp": GradedSharpReward,
    "format": FormatReward,
}


def get_reward_fn(name: str, **kwargs) -> RewardFunction:
    """Instantiate a reward function by name."""
    if name not in REWARD_REGISTRY:
        raise ValueError(f"Unknown reward function '{name}'. Available: {list(REWARD_REGISTRY.keys())}")
    return REWARD_REGISTRY[name](**kwargs)
