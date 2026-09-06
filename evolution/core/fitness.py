"""Fitness functions for evaluating evolved artifacts.

Uses LLM-as-judge with rubrics to score agent outputs.
Supports length penalties and multi-dimensional scoring.
"""

import dspy
from dataclasses import dataclass
from typing import Optional

from evolution.core.config import EvolutionConfig


@dataclass
class FitnessScore:
    """Multi-dimensional fitness score."""
    correctness: float = 0.0  # Did the agent produce correct output? (0-1)
    procedure_following: float = 0.0  # Did it follow the skill's procedure? (0-1)
    conciseness: float = 0.0  # Was it appropriately concise? (0-1)
    length_penalty: float = 0.0  # Penalty for being too verbose (0-1, 0 = no penalty)
    feedback: str = ""  # Textual feedback for GEPA's reflective analysis

    @property
    def composite(self) -> float:
        """Weighted composite score."""
        raw = (
            0.5 * self.correctness
            + 0.3 * self.procedure_following
            + 0.2 * self.conciseness
        )
        return max(0.0, raw - self.length_penalty)


class LLMJudge:
    """LLM-as-judge scorer with rubric-based evaluation.

    Scores agent outputs on multiple dimensions and provides
    textual feedback that GEPA can use for reflective mutation.
    """

    class JudgeSignature(dspy.Signature):
        """Evaluate an agent's response against an expected behavior rubric.

        Score the response on three dimensions (0.0 to 1.0 each):
        1. correctness: Did the response correctly address the task?
        2. procedure_following: Did it follow the expected approach/procedure?
        3. conciseness: Was it appropriately concise without omitting important info?

        Also provide specific, actionable feedback on what could be improved.
        """
        task_input: str = dspy.InputField(desc="The task the agent was given")
        expected_behavior: str = dspy.InputField(desc="Rubric describing what a good response looks like")
        agent_output: str = dspy.InputField(desc="The agent's actual response")
        skill_text: str = dspy.InputField(desc="The skill/instructions the agent was following")
        correctness: float = dspy.OutputField(desc="Score 0.0-1.0: Did the response correctly address the task?")
        procedure_following: float = dspy.OutputField(desc="Score 0.0-1.0: Did it follow the expected procedure?")
        conciseness: float = dspy.OutputField(desc="Score 0.0-1.0: Appropriately concise?")
        feedback: str = dspy.OutputField(desc="Specific, actionable feedback on what could be improved")

    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.judge = dspy.ChainOfThought(self.JudgeSignature)

    def score(
        self,
        task_input: str,
        expected_behavior: str,
        agent_output: str,
        skill_text: str,
        artifact_size: Optional[int] = None,
        max_size: Optional[int] = None,
    ) -> FitnessScore:
        """Score an agent output using LLM-as-judge."""

        lm = dspy.LM(self.config.eval_model)

        with dspy.context(lm=lm):
            result = self.judge(
                task_input=task_input,
                expected_behavior=expected_behavior,
                agent_output=agent_output,
                skill_text=skill_text,
            )

        # Parse scores (clamp to 0-1)
        correctness = _parse_score(result.correctness)
        procedure_following = _parse_score(result.procedure_following)
        conciseness = _parse_score(result.conciseness)

        # Length penalty
        length_penalty = 0.0
        if artifact_size is not None and max_size is not None:
            ratio = artifact_size / max_size
            if ratio > 0.9:
                # Penalty ramps from 0 at 90% to 0.3 at 100%+
                length_penalty = min(0.3, (ratio - 0.9) * 3.0)

        return FitnessScore(
            correctness=correctness,
            procedure_following=procedure_following,
            conciseness=conciseness,
            length_penalty=length_penalty,
            feedback=str(result.feedback),
        )


def skill_fitness_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None, force_judge: bool = False) -> float:
    """DSPy-compatible metric function for skill optimization.

    Hybrid scorer:
    - Default: fast keyword-overlap heuristic (cheap during optimization).
    - Every `metric_judge_every`-th call (configured via configure_metric):
      LLM-as-judge with the full rubric (correctness, procedure following,
      conciseness + feedback), so the optimizer receives real quality signal.
    - force_judge=True: always use LLM-as-judge (final holdout evaluation).

    Returns a float 0-1 score.
    """
    # The prediction should have an 'output' field with the agent's response
    agent_output = getattr(prediction, "output", "") or ""
    expected = getattr(example, "expected_behavior", "") or ""

    if not agent_output.strip():
        return 0.0

    config, judge_every = _METRIC_CONFIG or (None, 0)

    use_judge = force_judge
    if not use_judge and judge_every and judge_every > 0:
        _judge_counter[0] += 1
        use_judge = (_judge_counter[0] % judge_every) == 0

    if use_judge and config is not None:
        try:
            task = getattr(example, "task_input", "") or ""
            skill_text = getattr(example, "skill_text", "") or ""
            fs = LLMJudge(config).score(
                task_input=task,
                expected_behavior=expected,
                agent_output=agent_output,
                skill_text=skill_text,
            )
            return round(fs.composite, 4)
        except Exception:
            # Judge failure must never kill an optimization run —
            # fall back to the heuristic for this call.
            pass

    # Quick heuristic scoring (for speed during optimization)
    # Full LLM-as-judge scoring is expensive — use it selectively
    score = 0.5  # Base score for non-empty output

    # Check if key phrases from expected behavior appear
    expected_lower = expected.lower()
    output_lower = agent_output.lower()

    # Simple keyword overlap as a fast proxy
    expected_words = set(expected_lower.split())
    output_words = set(output_lower.split())
    if expected_words:
        overlap = len(expected_words & output_words) / len(expected_words)
        score = 0.3 + (0.7 * overlap)

    return min(1.0, max(0.0, score))


# Module-level metric configuration (set once at evolve start).
_METRIC_CONFIG = None  # (EvolutionConfig, judge_every)
_judge_counter = [0]


def configure_metric(config: EvolutionConfig, judge_every: int = 4) -> None:
    """Route every `judge_every`-th metric call through the LLM judge."""
    global _METRIC_CONFIG
    _METRIC_CONFIG = (config, max(0, judge_every))


def reset_metric_counter() -> None:
    """Reset the judge call counter (per optimization run)."""
    _judge_counter[0] = 0


def _parse_score(value) -> float:
    """Parse a score value, handling various LLM output formats."""
    if isinstance(value, (int, float)):
        return min(1.0, max(0.0, float(value)))
    try:
        return min(1.0, max(0.0, float(str(value).strip())))
    except (ValueError, TypeError):
        return 0.5  # Default to neutral on parse failure
