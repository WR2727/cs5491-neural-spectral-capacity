"""Specification-only FFN discounts for the archived Transformer family."""
import math

from nsc import capacity_components


ALPHA_GRID = (0.0, 0.25, 0.5, 1.0, 2.0)
# Both methods attain their highest development Spearman rho at alpha = 2.
ALPHA = 2.0


def preceding_fractions(layer_totals):
    """Normalize each exclusive prefix of layer capacities by full baseline NSC."""
    total = math.fsum(layer_totals)
    preceding = 0.0
    fractions = []
    for capacity in layer_totals:
        fractions.append(preceding / total)
        preceding += capacity
    return fractions


def discounted_score(components, fractions, alpha):
    """Keep attention capacity and discount only the two FFN projections."""
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    return math.fsum(
        attention + ffn / (1 + alpha * fraction)
        for (attention, ffn), fraction in zip(components, fractions)
    )


def score(config, alpha=ALPHA):
    """Sum attention and FFN capacity discounted by relative preceding capacity."""
    components = capacity_components(config)
    totals = [attention + ffn for attention, ffn in components]
    return discounted_score(components, preceding_fractions(totals), alpha)


def position_control(config, alpha=ALPHA):
    """Use the same discount, replacing relative capacity by layer position."""
    components = capacity_components(config)
    depth = len(components)
    fractions = [index / depth for index in range(depth)]
    return discounted_score(components, fractions, alpha)


def ablation(config):
    """Set alpha to zero to remove the FFN discount."""
    return score(config, alpha=0.0)
