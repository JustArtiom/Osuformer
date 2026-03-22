from ..config import ExperimentConfig
from .seq2seq import Seq2SeqModel
from .conformer import ConformerEncoder
from .transformer import TransformerDecoder


def build_model(
  config: ExperimentConfig, 
  vocab_size: int,
) -> Seq2SeqModel:
  
  enc_cfg = config.model.encoder
  encoder = ConformerEncoder(
    input_dim=config.audio.n_mels,
    d_model=enc_cfg.d_model,
    num_layers=enc_cfg.layers,
    num_heads=enc_cfg.heads,
    ffn_dim=enc_cfg.ffn_dim,
    conv_kernel=enc_cfg.conv_kernel,
    dropout=enc_cfg.dropout,
    positional_encoding="relative",
    max_relative_position=128,
    relative_style="bias",
    subsampling=True,
    conditioning_dim=2,  # song position: [start_frac, end_frac]
  )

  dec_cfg = config.model.decoder
  decoder = TransformerDecoder(
    vocab_size=vocab_size,
    d_model=dec_cfg.d_model,
    n_heads=dec_cfg.heads,
    d_ff=dec_cfg.ffn_dim,
    n_layers=dec_cfg.layers,
    dropout=dec_cfg.dropout,
    pad_id=0,
  )

  return Seq2SeqModel(
      encoder=encoder,
      decoder=decoder,
  )