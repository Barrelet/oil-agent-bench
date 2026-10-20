"""What an agent returns each day, and the strict answer format the LLMs must follow.

One decision per day:
  position  : "long" | "short" | "flat" — the position wanted after the next open
  p_up      : the agent's probability that the NEXT close is above today's close.
              Scored on its own (Brier score, calibration), with no threshold attached,
              which fixes the flaw found in September (models wrote the smallest
              confidence the rules accepted).
Asking for the position you already hold is a hold. Players never set a stop: the game places
the same disaster stop on every position (rules.STOP_ATR_MULT x ATR from entry).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

Position = Literal["long", "short", "flat"]


@dataclass
class Decision:
    position: str                    # long | short | flat
    p_up: float | None = None        # probability next close > today's close
    thesis: str = ""
    reasoning: str = ""
    invalid: str = ""                # non-empty when the answer could not be used (treated as a hold)

    @classmethod
    def hold(cls, current: str, why: str = "") -> "Decision":
        return cls(position=current, invalid=why)


class LLMAnswer(BaseModel):
    """Strict JSON contract for the language models."""
    reasoning: str = Field(description="<= 80 words: what you checked and why you chose this")
    p_up: float = Field(ge=0.0, le=1.0, description="probability that the next close is ABOVE today's close")
    position: Position
    thesis: str = Field(default="", description="<= 25 words")


def parse_llm(obj: dict | str) -> tuple[LLMAnswer | None, str | None]:
    try:
        a = LLMAnswer.model_validate_json(obj) if isinstance(obj, str) else LLMAnswer.model_validate(obj)
        return a, None
    except ValidationError as e:
        return None, str(e)


def json_schema() -> dict:
    return LLMAnswer.model_json_schema()
