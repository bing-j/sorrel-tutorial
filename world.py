# begin imports
from sorrel.worlds import Gridworld
from metrics_collector import StaghuntMetricsCollector
# end imports

# begin staghunt
class StaghuntWorld(Gridworld):

    def __init__(self, config, default_entity):
        super().__init__(
            height=config.world.height,
            width=config.world.width,
            layers=4,
            default_entity=default_entity
        )

        self.stag_value = config.world.stag_value
        self.hare_value = config.world.hare_value
        self.spawn_prob = config.world.spawn_prob
        self.beam_radius = config.world.beam_radius
        self.num_of_agents = config.world.num_of_agents

        self.metrics_collector = StaghuntMetricsCollector()
