# begin imports
from sorrel.worlds import Gridworld
# end imports

# begin staghunt
class StaghuntWorld(Gridworld):

    def __init__(self, config, default_entity):
        super().__init__(
            height=11,
            width=11,
            layers=3,
            default_entity=default_entity
        )

        self.stag_value = config.world.stag_value
        self.hare_value = config.world.hare_value
        self.spawn_prob = config.world.spawn_prob
