# begin imports
# general imports
from typing import Optional
import random

import numpy as np
import torch
from typing_extensions import override

from sorrel.action.action_spec import ActionSpec
from sorrel.environment import Environment
from sorrel.utils.turn_stats import TurnStats

# imports from our example
from agents import StaghuntAgent
from entities import EmptyEntity, Sand, SpawnTile, Wall
from metrics_collector import StaghuntMetricsCollector
from world import StaghuntWorld

# sorrel imports
from sorrel.models.pytorch import PyTorchIQN
from sorrel.observation.observation_spec import OneHotObservationSpec

# end imports

# begin staghunt environment
class StaghuntEnv(Environment[StaghuntWorld]):
    """The experiment for staghunt."""

    def __init__(self, world: StaghuntWorld, config: dict) -> None:
        super().__init__(world, config)

    # end constructor

    def setup_agents(self):
        """Create the agents for this experiment and assign them to self.agents.

        Requires self.config.model.agent_vision_radius to be defined.
        """
        agent_num = self.world.num_of_agents
        agents = []

        # create the observation spec
        entity_list = [
            "EmptyEntity",
            "Wall",
            "Stag",
            "Hare",
            "StaghuntAgent",
            "Beam",
        ]

        for agent_id in range(agent_num):

            observation_spec_class = OneHotObservationSpec

            observation_spec = observation_spec_class(
                entity_list,
                full_view=False,
                vision_radius=3,
            )

            # create the action spec
            action_spec = ActionSpec(["up", "down", "left", "right", "zap"])

            # create the model
            model = PyTorchIQN(
                input_size=observation_spec.input_size,
                action_space=action_spec.n_actions,
                layer_size=250,
                epsilon=0.6,
                device=self.config.experiment.device,
                seed=torch.random.seed(),
                n_frames=5,
                n_step=3,
                sync_freq=200,
                model_update_freq=4,
                batch_size=64,
                memory_size=1024,
                LR=0.00025,
                TAU=0.001,
                GAMMA=0.99,
                n_quantiles=12,
            )

            agents.append(
                StaghuntAgent(
                    observation_spec=observation_spec,
                    action_spec=action_spec,
                    model=model,
                    agent_id=agent_id,
                )
            )

        self.agents = agents

    def populate_environment(self):
        """Populate the staghunt world by creating walls, then randomly spawning the
        agents.

        Note that self.world.map is already created with the specified dimensions, and
        every space is filled with EmptyEntity, as part of super().__init__() when this
        experiment is constructed.
        """
        valid_spawn_locations = []

        for index in np.ndindex(self.world.map.shape):
            y, x, z = index
            if (y in [0, self.world.height - 1] or x in [0, self.world.width - 1]) and (
                z == 2
            ):
                # Add walls around the edge of the world (when indices are first or last)
                self.world.add(index, Wall())
            elif z == 0:  # if location is on the bottom layer, put sand there
                self.world.add(index, Sand())
            elif z == 1:  # level 1: objects spawning
                self.world.add(index, SpawnTile())
            elif z == 2:  # level 2: valid spawn location
                valid_spawn_locations.append(index)
            elif z == 3:  # beam layer
                self.world.add(index, EmptyEntity())

        # spawn the agents
        agent_locations = random.sample(valid_spawn_locations, k=len(self.agents))

        for loc, agent in zip(agent_locations, self.agents):
            self.world.add(loc, agent)

    @override
    def collect_turn_stats(self, epoch: int) -> TurnStats:
        """Augment the base per-turn snapshot with staghunt-specific event counts."""
        stats = super().collect_turn_stats(epoch)
        stats.extra.update(self.world.metrics_collector.create_world_specific_stats_dict())
        return stats

    @override
    def on_turn_end(self, stats:TurnStats) -> None:
        super().on_turn_end(stats)
        self.world.metrics_collector.reset_turn()

    @override
    def aggregate_epoch_stats(self) -> dict[str, float]:
        """Sum per-turn staghunt metrics (global and per-agent) across the epoch."""
        base = super().aggregate_epoch_stats()
        if not self._epoch_turn_stats:
            return base

        scalar_keys = [
            "attacks_to_hares",
            "attacks_to_stags",
            "resources_defeated",
            "stags_defeated",
            "hares_defeated",
        ]
        by_agent_keys = [
            "attacks_to_hares_by_agent",
            "attacks_to_stags_by_agent",
            "resources_defeated_by_agent",
            "stags_defeated_by_agent",
            "hares_defeated_by_agent",
        ]

        totals = {k: 0 for k in scalar_keys}
        by_agent_totals: dict[str, dict[str, int]] = {k: {} for k in by_agent_keys}

        for ts in self._epoch_turn_stats:
            for k in scalar_keys:
                totals[k] += ts.extra.get(k, 0)
            for k in by_agent_keys:
                for agent_key, count in ts.extra.get(k, {}).items():
                    by_agent_totals[k][agent_key] = (
                        by_agent_totals[k].get(agent_key, 0) + count
                    )

        total_attacks = totals["attacks_to_hares"] + totals["attacks_to_stags"]
        ratios = {
            "hare_attack_ratio": (
                totals["attacks_to_hares"] / total_attacks if total_attacks else 0.0
            ),
            "stag_attack_ratio": (
                totals["attacks_to_stags"] / total_attacks if total_attacks else 0.0
            ),
        }

        return {
            **base,
            **{f"total_{k}": v for k, v in totals.items()},
            **{f"total_{k}": v for k, v in by_agent_totals.items()},
            **ratios,
        }