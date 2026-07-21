import hydra
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from src.utils import set_seed
from src.runner import get_trainer
from src.trainer import lock as work_lock


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
    if trainer is None:
        # Another worker owns this config, or it's already completed.
        # No wandb.init happened; exit cleanly so ray/hydra treat the task
        # as a no-op success rather than a crash.
        return

    trainer.train()
    # Success path: mark done BEFORE finish + release. Order matters so that
    # a crash between the two doesn't leave a lock claimable + config
    # unmarked (which would let another worker start over from checkpoint —
    # harmless but wasteful). `done` sitting on disk with a released lock is
    # the correct terminal state.
    if trainer.checkpoint_path is not None:
        work_lock.mark_completed(trainer.checkpoint_path.parent)
    # Explicit `wandb.finish()` on the success path only. Without it, Runai's
    # pod teardown races wandb's atexit flush and successful runs get marked
    # "crashed". Deliberately NOT in `finally`: on a real exception we want
    # wandb to see the abrupt exit and mark the run "crashed" — the accurate
    # state — instead of masking the failure as a clean finish.
    if wandb.run is not None:
        wandb.finish()
    work_lock.release()


if __name__ == "__main__":
    train()
