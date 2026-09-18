"""Metrics collection system for Staghunt environment.

This module provides a pure accumulator for domain-specific event counts
(attacks, resource defeats). It has no knowledge of Logger/TensorBoard —
StaghuntEnv reads from it in collect_turn_stats/aggregate_epoch_stats and
resets it each turn. All actual logging is handled by the base Environment
class and the active Logger.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from agents import StaghuntAgent

class StaghuntMetricsCollector:
    """Accumulates raw per-turn event counts for the staghunt environment."""

    def __init__(self, num_agents: int = 0):
        self.num_agents = num_agents
        self.reset_turn()

    def reset_turn(self) -> None:
        """Reset all counters. Call at the end of every turn, after
        StaghuntEnv.collect_turn_stats has read the values into TurnStats.extra.
        """
        # global (turn-level) counters
        self.attacks_to_hares = 0
        self.attacks_to_stags = 0
        self.resources_defeated = 0
        self.stags_defeated = 0
        self.hares_defeated = 0

        # make sure that agents with no events this turn still add a 0 for tensorboard logging
        def _zero_filled() -> dict[int, int]:
            return defaultdict(int, {i: 0 for i in range(self.num_agents)})

        self.attacks_to_hares_by_agent: dict[int, int] = _zero_filled()
        self.attacks_to_stags_by_agent: dict[int, int] = _zero_filled()
        self.resources_defeated_by_agent: dict[int, int] = _zero_filled()
        self.stags_defeated_by_agent: dict[int, int] = _zero_filled()
        self.hares_defeated_by_agent: dict[int, int] = _zero_filled()

    def collect_attack_metrics(self, agent: StaghuntAgent, target_type: str) -> None:
        """Record an attack action.

        Args:
            agent: The attacking agent.
            target_type: Type of target ("hare", "stag", etc.).
        """
        agent_id = agent.agent_id
        t = target_type.lower()
        if t == "hare":
            self.attacks_to_hares += 1
            self.attacks_to_hares_by_agent[agent_id] += 1
        elif t == "stag":
            self.attacks_to_stags += 1
            self.attacks_to_stags_by_agent[agent_id] += 1

    def collect_resource_defeat_metrics(
        self, agent: StaghuntAgent, resource_type: Optional[str] = None
    ) -> None:
        """Record a resource (stag/hare) being defeated.

        Args:
            agent: The agent who defeated the resource.
            resource_type: Type of resource defeated ("stag" or "hare").
        """
        agent_id = agent.agent_id
        self.resources_defeated += 1
        self.resources_defeated_by_agent[agent_id] += 1

        if resource_type:
            rt = resource_type.lower()
            if rt == "stag":
                self.stags_defeated += 1
                self.stags_defeated_by_agent[agent_id] += 1
            elif rt == "hare":
                self.hares_defeated += 1
                self.hares_defeated_by_agent[agent_id] += 1

    def create_world_specific_stats_dict(self, agent_id_prefix: str = "agent_") -> dict:
        """Package this turn's counters into a TurnStats.extra-compatible dict.

        Global counts become scalar entries; per-agent counts become dict
        entries so TensorboardLogger renders them as grouped multi-line
        charts via add_scalars.
        """
        def by_agent(d: dict[int, int]) -> dict[str, int]:
            return {f"{agent_id_prefix}{k}": v for k, v in d.items()}

        return {
            "attacks_to_hares": self.attacks_to_hares,
            "attacks_to_stags": self.attacks_to_stags,
            "resources_defeated": self.resources_defeated,
            "stags_defeated": self.stags_defeated,
            "hares_defeated": self.hares_defeated,
            "attacks_to_hares_by_agent": by_agent(self.attacks_to_hares_by_agent),
            "attacks_to_stags_by_agent": by_agent(self.attacks_to_stags_by_agent),
            "resources_defeated_by_agent": by_agent(self.resources_defeated_by_agent),
            "stags_defeated_by_agent": by_agent(self.stags_defeated_by_agent),
            "hares_defeated_by_agent": by_agent(self.hares_defeated_by_agent),
        }