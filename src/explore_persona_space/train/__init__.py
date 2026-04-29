"""Training pipeline package.

Re-exports public symbols so that both of these import styles work:
    from explore_persona_space.train.trainer import train_phase
    from explore_persona_space.train.distributed import run_distributed_pipeline
"""

from explore_persona_space.train.distributed import run_distributed_pipeline  # noqa: F401
