# for configs
from pathlib import Path
import hydra

# Sorrel imports
from entities import EmptyEntity
from env import StaghuntEnv
from world import StaghuntWorld
from sorrel.utils.logging import Logger, TensorboardLogger

@hydra.main(version_base=None, config_path=".", config_name="cfg")
def main(config):
    env = StaghuntWorld(config=config, default_entity=EmptyEntity())
    experiment = StaghuntEnv(env, config)
    logger = TensorboardLogger.from_config(config)
    experiment.run_experiment(output_dir=Path(__file__).parent / "./data", logger=logger)


# begin main
if __name__ == "__main__":
    main()