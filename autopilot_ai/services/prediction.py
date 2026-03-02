"""
services/prediction.py — Saturation forecasting and issue ranking.

This module provides lightweight, dependency-free numerical methods so that
the API layer can give users forward-looking answers:

  predict_saturation(samples, threshold, lookahead_minutes)
      → time-to-threshold estimate using least-squares linear regression
      → returns SaturationPrediction (eta_minutes, confidence, trend_slope)

  rank_issues(insights)
      → composite score = urgency_weight × confidence × (1 + cost_factor)
      → returns insights sorted descending priority

No external ML dependencies — uses only Python builtins and simple maths.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Sequence

from autopilot_ai.core.logging import get_logger
from autopilot_ai.models.insights import Insight, Urgency

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_URGENCY_WEIGHTS: dict[Urgency, float] = {
    Urgency.IMMEDIATE: 4.0,
    Urgency.HIGH:      3.0,
    Urgency.MEDIUM:    2.0,
    Urgency.LOW:       1.0,
}

# Maximum monthly cost impact used to normalise cost factor onto [0, 1].
# Anything above this cap is treated as equally important.
_COST_NORMALISATION_INR = 500_000.0    # ₹5 lakh / month


# ── Data classes ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SaturationPrediction:
    """Result of a saturation prediction run."""

    # Minutes until the metric is predicted to reach `threshold`.
    # None if the metric is not trending toward the threshold.
    eta_minutes: float | None

    # Slope of the fitted line (units per minute).
    trend_slope: float

    # R² of the linear fit, used as a proxy for prediction confidence.
    r_squared: float

    # Predicted datetime of threshold breach (None if eta_minutes is None).
    predicted_at: datetime | None

    # The threshold that was used.
    threshold: float

    # Current (latest) value.
    current_value: float

    @property
    def confidence(self) -> float:
        """Confidence in the prediction on [0, 1], clamped from R²."""
        return max(0.0, min(1.0, self.r_squared))

    @property
    def is_approaching(self) -> bool:
        """True if the metric is trending toward (not away from) the threshold."""
        return self.eta_minutes is not None and self.eta_minutes > 0


@dataclass
class RankedInsight:
    """An Insight paired with its computed priority score."""

    insight: Insight
    score: float = field(default=0.0)

    # Breakdown for transparency / debugging
    urgency_weight: float = field(default=0.0)
    confidence: float = field(default=0.0)
    cost_factor: float = field(default=0.0)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _linear_regression(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """
    Simple OLS linear regression: y = slope × x + intercept.

    Returns (slope, intercept, r_squared).
    r_squared = 0 if all y-values are identical (zero variance).
    """
    n = len(xs)
    if n < 2:
        return 0.0, ys[0] if ys else 0.0, 0.0

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    ss_xy = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
    ss_xx = sum((xs[i] - mean_x) ** 2 for i in range(n))
    ss_yy = sum((ys[i] - mean_y) ** 2 for i in range(n))

    if ss_xx == 0:
        return 0.0, mean_y, 0.0

    slope = ss_xy / ss_xx
    intercept = mean_y - slope * mean_x

    r_squared = (ss_xy ** 2 / (ss_xx * ss_yy)) if ss_yy > 0 else 1.0
    return slope, intercept, r_squared


def _cost_factor(insight: Insight) -> float:
    """
    Normalise the highest monthly cost impact across all recommendations to [0, 1].
    Returns 0.0 if no recommendations have a cost impact attached.
    """
    best = 0.0
    for rec in insight.recommendations:
        if rec.cost_impact and rec.cost_impact.monthly_inr > best:
            best = rec.cost_impact.monthly_inr
    return min(1.0, best / _COST_NORMALISATION_INR) if best > 0 else 0.0


# ── PredictionService ─────────────────────────────────────────────────────────
class PredictionService:
    """
    Provides two operations:
      1. predict_saturation — when will a metric cross a threshold?
      2. rank_issues        — sort Insights by composite urgency score.
    """

    # ── Saturation prediction ─────────────────────────────────────────────

    def predict_saturation(
        self,
        samples: Sequence[float],
        threshold: float,
        *,
        sample_interval_minutes: float = 1.0,
        lookahead_minutes: float = 60.0,
    ) -> SaturationPrediction:
        """
        Fit a linear trend to `samples` and estimate when the metric will
        reach `threshold`.

        Args:
            samples: Time-ordered metric values (oldest first). At least 2
                     values are required for a meaningful prediction.
            threshold: The saturation or alert threshold value.
            sample_interval_minutes: Minutes between consecutive samples.
            lookahead_minutes: Maximum ETA to consider; predictions beyond
                               this window are treated as "not approaching".

        Returns:
            SaturationPrediction with eta_minutes = None if the metric is
            not trending toward the threshold within the lookahead window.
        """
        samples = list(samples)
        if not samples:
            logger.warning("prediction_no_samples")
            return SaturationPrediction(
                eta_minutes=None,
                trend_slope=0.0,
                r_squared=0.0,
                predicted_at=None,
                threshold=threshold,
                current_value=0.0,
            )

        xs = [i * sample_interval_minutes for i in range(len(samples))]
        slope, intercept, r_squared = _linear_regression(xs, samples)
        current = samples[-1]
        current_x = xs[-1]

        # Solve for x where slope*x + intercept == threshold
        eta_minutes: float | None = None
        predicted_at: datetime | None = None

        if slope != 0:
            x_threshold = (threshold - intercept) / slope
            # eta is relative to the last sample
            eta_relative = x_threshold - current_x
            if 0 < eta_relative <= lookahead_minutes:
                eta_minutes = eta_relative
                predicted_at = datetime.now(tz=timezone.utc) + timedelta(minutes=eta_relative)

        logger.debug(
            "prediction_saturation",
            n_samples=len(samples),
            current=round(current, 3),
            threshold=threshold,
            slope=round(slope, 5),
            r_squared=round(r_squared, 4),
            eta_minutes=round(eta_minutes, 1) if eta_minutes else None,
        )

        return SaturationPrediction(
            eta_minutes=eta_minutes,
            trend_slope=slope,
            r_squared=r_squared,
            predicted_at=predicted_at,
            threshold=threshold,
            current_value=current,
        )

    def estimate_time_to_threshold(
        self,
        values: Sequence[float],
        threshold: float,
        sample_interval_minutes: float = 1.0,
    ) -> float | None:
        """
        Convenience wrapper: returns eta_minutes (float) or None.
        """
        pred = self.predict_saturation(
            values,
            threshold,
            sample_interval_minutes=sample_interval_minutes,
        )
        return pred.eta_minutes

    # ── Issue ranking ─────────────────────────────────────────────────────

    def rank_issues(self, insights: Sequence[Insight]) -> list[RankedInsight]:
        """
        Rank Insights by a composite priority score:

            score = urgency_weight × confidence × (1 + cost_factor)

        Higher scores surface first. Insights with equal scores keep their
        original order (stable sort).
        """
        ranked: list[RankedInsight] = []

        for insight in insights:
            uw = _URGENCY_WEIGHTS.get(insight.urgency, 1.0)
            cf = _cost_factor(insight)
            score = uw * insight.confidence * (1.0 + cf)

            ranked.append(RankedInsight(
                insight=insight,
                score=round(score, 4),
                urgency_weight=uw,
                confidence=insight.confidence,
                cost_factor=cf,
            ))

        ranked.sort(key=lambda r: r.score, reverse=True)

        logger.debug("prediction_ranked_issues", count=len(ranked))
        return ranked

    def top_n(self, insights: Sequence[Insight], n: int = 5) -> list[Insight]:
        """Return the top-n highest-priority Insights."""
        return [r.insight for r in self.rank_issues(insights)[:n]]


# Module-level singleton
prediction_service = PredictionService()
