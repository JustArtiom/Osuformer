from typing import TypedDict


class LrSchedulerConfig(TypedDict):
    enabled: bool
    type: str
    min_lr: float


class TrainingConfig(TypedDict):
    batch_size: int
    learning_rate: float
    warmup_steps: int
    max_steps: int
    grad_clip: float
    weight_decay: float
    save_every_steps: int
    keep_last_checkpoint: bool
    keep_best_checkpoint: bool
    best_metric: str
    lr_scheduler: LrSchedulerConfig
