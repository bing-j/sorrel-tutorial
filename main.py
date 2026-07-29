# for configs
from pathlib import Path
import hydra

# Sorrel imports
from entities import Empty
from env import StaghuntEnv
from world import StaghuntWorld
from sorrel.utils.logging import Logger, TensorboardLogger

@hydra.main(version_base=None, config_path=".", config_name="cfg")
def main(config):
    # Future: integrate additonal parsed arguments into the configuration path?
    env = StaghuntWorld(config=config, default_entity=EmptyEntity())
    experiment = StaghuntEnv(env, config)
    experiment.run_experiment(output_dir=Path(__file__).parent / "./data")


# begin main
if __name__ == "__main__":
    main()