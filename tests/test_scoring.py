import pytest

from src.decision.scoring import DecisionWeights, TierProperties, score_tier
from src.storage.tier import Tier


def test_score_exposes_weighted_contributions_and_candidate_latency():
    weights = DecisionWeights(access=0.4, semantic=0.2, latency=0.2, migration=0.1, resource=0.1)
    source = TierProperties(0.8, 1000, relative_cost=0.2, migration_cost=1.0)
    cache = TierProperties(1.0, 1000, relative_cost=0.4, migration_cost=1.0)

    score = score_tier(0.5, 0.75, 100, Tier.SSD, Tier.CACHE, source, cache, 0.5, weights, 1000)

    assert score.access == pytest.approx(0.2)
    assert score.semantic == pytest.approx(0.15)
    assert score.latency == pytest.approx(0.2)
    assert score.migration == pytest.approx(-0.01)
    assert score.resource == pytest.approx(-0.045)
    assert score.total == pytest.approx(0.495)


def test_current_tier_has_no_migration_penalty_and_invalid_probability_fails():
    weights = DecisionWeights(1, 1, 1, 1, 1)
    properties = TierProperties(0.5, 100)
    score = score_tier(0.5, 0.5, 10, Tier.SSD, Tier.SSD, properties, properties, 0.1, weights)

    assert score.migration == 0
    with pytest.raises(ValueError, match="access_probability"):
        score_tier(1.1, 0.5, 10, Tier.SSD, Tier.SSD, properties, properties, 0.1, weights)