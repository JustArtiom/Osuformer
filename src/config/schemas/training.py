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
    timing_jitter_bins: int
    lr_scheduler: LrSchedulerConfig
    encoder_lr_scale: float = 1.0
    rhythm_loss_weight: float = 3.0
    slider_loss_weight: float = 1.0
    cfg_dropout_prob: float = 0.1
    aux_star_weight: float = 0.05
    aux_descriptor_weight: float = 0.05
    aux_density_weight: float = 0.05
    aux_warmup_steps: int = 2000
    z_loss_weight: float = 1.0e-4
