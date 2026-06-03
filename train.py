import hydra
from omegaconf import DictConfig, OmegaConf

from src.utils import set_seed
from src.runner import get_trainer


@hydra.main(version_base=None, config_path="conf", config_name="train")
def train(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # set the seed
    set_seed(cfg.misc.seed)

    # initialize objects
    trainer = get_trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    train()
