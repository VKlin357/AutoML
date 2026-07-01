"""
Regularized Evolution loop (Real et al., AAAI 2019), parameterised by a
pluggable Mutator / Crossover. The LLM agent is one such mutator; a
random one is the baseline ablation.

Key knobs:

- ``population_size``      P  : how many individuals are kept alive
- ``tournament_size``      S  : how many randomly sampled individuals
                                from the population are shown to the
                                mutator for it to pick a parent from
- ``aging``: True          : on each step we evict the *oldest* member
                              (not the worst). This prevents mode-collapse,
                              which is critical when the mutator is an
                              LLM that tends to fixate on its first hit.
- ``crossover_p``               probability of doing crossover instead of
                              mutation (only triggers if there are >= 2
                              individuals).

Each individual stores: cfg, primary, trial_id, birth_step.

This module does *no* training itself — the orchestrator owns the
training loop and just calls ``Population.add`` after each evaluated
trial.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .search_space import (
    config_distance,
    crossover as random_crossover,
    mutate_random,
    sample_random_config,
    validate_config,
)


@dataclass
class Individual:
    cfg: Dict[str, Any]
    primary: float
    trial_id: int
    birth_step: int


@dataclass
class Population:
    population_size: int = 25
    tournament_size: int = 5
    aging: bool = True
    members: List[Individual] = field(default_factory=list)
    step: int = 0

    def add(self, cfg: Dict[str, Any], primary: float, trial_id: int) -> None:
        ind = Individual(cfg=cfg, primary=primary, trial_id=trial_id, birth_step=self.step)
        self.step += 1
        self.members.append(ind)
        if len(self.members) > self.population_size:
            self.evict_one()

    def evict_one(self) -> Optional[Individual]:
        if not self.members:
            return None
        if self.aging:
            # remove oldest (smallest birth_step)
            idx = min(range(len(self.members)), key=lambda i: self.members[i].birth_step)
        else:
            # standard "kill worst" evolution
            idx = min(range(len(self.members)), key=lambda i: self.members[i].primary)
        return self.members.pop(idx)

    def best(self) -> Optional[Individual]:
        return max(self.members, key=lambda x: x.primary) if self.members else None

    def tournament(self, rng: random.Random) -> List[Individual]:
        if not self.members:
            return []
        k = min(self.tournament_size, len(self.members))
        return rng.sample(self.members, k=k)

    def diverse_subsample(self, k: int) -> List[Individual]:
        """Greedy diversity selection from the current pop (for prompts)."""
        if len(self.members) <= k:
            return list(self.members)
        chosen = [max(self.members, key=lambda x: x.primary)]
        remaining = [m for m in self.members if m is not chosen[0]]
        while len(chosen) < k and remaining:
            # pick the member with the largest min-distance to the chosen set
            def min_dist(m):
                return min(config_distance(m.cfg, c.cfg) for c in chosen)
            nxt = max(remaining, key=min_dist)
            chosen.append(nxt)
            remaining.remove(nxt)
        return chosen


# ---------------------------------------------------------------------------
# Step interface
# ---------------------------------------------------------------------------

# A mutator takes (parent_cfg, history, rng) -> child_cfg
Mutator   = Callable[[Dict[str, Any], List[Individual], random.Random], Dict[str, Any]]
# A crossover takes (parent_a_cfg, parent_b_cfg, history, rng) -> child_cfg
Crossover = Callable[[Dict[str, Any], Dict[str, Any], List[Individual], random.Random], Dict[str, Any]]


def random_mutator(parent: Dict[str, Any], history: List[Individual], rng: random.Random) -> Dict[str, Any]:
    return validate_config(mutate_random(parent, rng, n_changes=rng.choice([1, 1, 1, 2])))


def random_crossover_op(a: Dict[str, Any], b: Dict[str, Any],
                        history: List[Individual], rng: random.Random) -> Dict[str, Any]:
    return validate_config(random_crossover(a, b, rng))


def cold_start_random(n: int, rng: random.Random) -> List[Dict[str, Any]]:
    """Diverse random initialisation: one config per arch family, then
    rest sampled freely."""
    from .search_space import ARCH_FAMILIES
    cfgs = [validate_config(sample_random_config(rng, family=fam)) for fam in ARCH_FAMILIES]
    while len(cfgs) < n:
        cfgs.append(validate_config(sample_random_config(rng)))
    return cfgs[:n]


# ---------------------------------------------------------------------------
# Single evolution step (parent selection -> child proposal)
# ---------------------------------------------------------------------------

def propose_next_config(
    pop: Population,
    *,
    mutator: Mutator,
    crossover_op: Crossover,
    rng: random.Random,
    crossover_p: float = 0.3,
) -> Tuple[Dict[str, Any], str]:
    """Pick parent(s) by tournament; return (child_cfg, op_name)."""
    if not pop.members:
        # bootstrap if pop empty
        return validate_config(sample_random_config(rng)), "bootstrap"
    if len(pop.members) >= 2 and rng.random() < crossover_p:
        a, b = rng.sample(pop.members, k=2)
        return crossover_op(a.cfg, b.cfg, list(pop.members), rng), "crossover"
    tourn = pop.tournament(rng)
    parent = max(tourn, key=lambda x: x.primary)
    return mutator(parent.cfg, list(pop.members), rng), "mutate"
