from dataclasses import dataclass


@dataclass
class LrSchedulerConfig:
    enabled: bool
    type: str
    min_lr: float


@dataclass
class TrainingConfig:
    run_name: str
    batch_size: int
    grad_accum_steps: int
    learning_rate: float
    warmup_steps: int
    max_steps: int
    grad_clip: float
    weight_decay: float
    save_every_steps: int
    log_every_steps: int
    val_every_steps: int
    keep_last_n_checkpoints: int
    keep_best_checkpoint: bool
    best_metric: str
    num_workers: int
    prefetch_factor: int
    pin_memory: bool
    precision: str
    seed: int
    resume: bool
    cache_name: str
    cache_preload: bool
    history_event_count: int
    max_decoder_len: int
    lr_scheduler: LrSchedulerConfig
