import hydra
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from src.utils import set_seed
from src.runner import get_trainer


@hydra.main(version_base=None, config_path="conf", config_name="train")
def train(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # Diagnostic: verify the checkpoint path resolves to a stable, absolute
    # location that survives across launches. Under the ray launcher,
    # `hydra.runtime.cwd` may point at a ray-owned tmp dir rather than the
    # project root, which would silently defeat resume.
    print(f"[debug] hydra.runtime.cwd = {HydraConfig.get().runtime.cwd}")
    print(f"[debug] cfg.misc.checkpoint.root = {cfg.misc.checkpoint.root}")

    set_seed(cfg.misc.seed)

    trainer = get_trainer(cfg)
    trainer.train()
    # Explicit `wandb.finish()` on the success path only. Without it, Runai's
    # pod teardown races wandb's atexit flush and successful runs get marked
    # "crashed". Deliberately NOT in `finally`: on a real exception we want
    # wandb to see the abrupt exit and mark the run "crashed" — the accurate
    # state — instead of masking the failure as a clean finish.
    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    train()
