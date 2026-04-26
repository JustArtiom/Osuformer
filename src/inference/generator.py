from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from tqdm.auto import tqdm

from src.config.schemas.audio import AudioConfig
from src.config.schemas.tokenizer import TokenizerConfig
from src.model import Osuformer
from src.osu_tokenizer import Event, EventType, SpecialToken, Vocab

from .grammar import GrammarState
from .prompt import GenerationPrompt, build_conditioning_tokens
from .sampler import SamplingConfig, sample_next_token


@dataclass
class GenerationResult:
    events: list[Event]
    window_starts_ms: list[float]


class WindowGenerator:
    def __init__(
        self,
        model: Osuformer,
        vocab: Vocab,
        tokenizer_cfg: TokenizerConfig,
        audio_cfg: AudioConfig,
        device: torch.device,
        max_decoder_len: int,
        history_event_count: int,
        sampling: SamplingConfig,
    ):
        self.model = model
        self.vocab = vocab
        self.tokenizer_cfg = tokenizer_cfg
        self.audio_cfg = audio_cfg
        self.device = device
        self.max_decoder_len = max_decoder_len
        self.history_event_count = history_event_count
        self.sampling = sampling
        self._window_total_ms = tokenizer_cfg.context_ms + tokenizer_cfg.generate_ms + tokenizer_cfg.lookahead_ms
        self._window_bins = self._window_total_ms // tokenizer_cfg.dt_bin_ms
        self._context_bins = tokenizer_cfg.context_ms // tokenizer_cfg.dt_bin_ms
        self._frames_per_window = int(self._window_total_ms / audio_cfg.hop_ms)
        abs_start, abs_end = vocab.token_range(EventType.ABS_TIME)
        self._abs_start = abs_start
        self._abs_end = abs_end
        self._max_rel_bin = vocab.range_for(EventType.REL_TIME).max_value
        self._vocab_out = vocab.vocab_size_out
        order = [er.type for er in vocab.output_ranges] + [er.type for er in vocab.input_ranges]
        self._type_order = order
        self._bias_vector = self._build_bias_vector()

    def _build_bias_vector(self) -> torch.Tensor:
        bias = torch.zeros(self._vocab_out)
        for event_type, value in self.sampling.event_bias.items():
            start, end = self.vocab.token_range(event_type)
            bias[start:end] = value
        return bias

    @torch.no_grad()
    def generate(
        self,
        mel: np.ndarray,
        prompt: GenerationPrompt,
        song_duration_ms: float,
        show_progress: bool = True,
    ) -> GenerationResult:
        self.model.eval()
        events: list[tuple[int, list[Event]]] = []
        cond_tokens = build_conditioning_tokens(prompt, self.vocab, self.tokenizer_cfg)
        window_starts: list[float] = []
        window_start_ms = 0.0
        total_windows = max(1, math.ceil(song_duration_ms / self.tokenizer_cfg.generate_ms))
        pbar = tqdm(total=total_windows, desc="generating", unit="window", disable=not show_progress)
        while window_start_ms < song_duration_ms:
            window_starts.append(window_start_ms)
            window_events = self._generate_window(
                mel=mel,
                prompt_conditioning_tokens=cond_tokens,
                window_start_ms=window_start_ms,
                history_groups=events,
            )
            events.extend(window_events)
            window_start_ms += self.tokenizer_cfg.generate_ms
            pbar.set_postfix(events=len(events), refresh=False)
            pbar.update(1)
        pbar.close()
        max_abs_bin = int(round(song_duration_ms / self.tokenizer_cfg.dt_bin_ms))
        events = [(raw_abs, group) for raw_abs, group in events if raw_abs < max_abs_bin]
        flat: list[Event] = []
        for raw_abs, group in events:
            flat.append(Event(EventType.ABS_TIME, raw_abs))
            flat.extend(group)
        return GenerationResult(events=flat, window_starts_ms=window_starts)

    def _generate_window(
        self,
        mel: np.ndarray,
        prompt_conditioning_tokens: list[int],
        window_start_ms: float,
        history_groups: list[tuple[int, list[Event]]],
    ) -> list[tuple[int, list[Event]]]:
        mel_slice = self._slice_mel(mel, window_start_ms)
        mel_tensor = torch.from_numpy(mel_slice.astype(np.float32)).unsqueeze(0).to(self.device)
        memory = self.model.encode(mel_tensor)
        grammar = GrammarState(self.vocab)

        suffix: list[int] = []
        last_raw_bin: int | None = None
        history = history_groups[-self.history_event_count :] if self.history_event_count > 0 else []
        for raw_abs, group in history:
            rel = self._rel(last_raw_bin, raw_abs)
            suffix.append(int(SpecialToken.TIME_ABS_NULL))
            suffix.append(self.vocab.encode_event(Event(EventType.REL_TIME, rel)))
            for ev in group:
                suffix.append(self.vocab.encode_event(ev))
            last_raw_bin = raw_abs
        suffix.append(int(SpecialToken.HISTORY_END))
        suffix.append(int(SpecialToken.SOS))

        cond_tokens: list[int] = list(prompt_conditioning_tokens) + suffix
        guidance = self.sampling.guidance_scale
        use_cfg = abs(guidance - 1.0) > 1e-6
        uncond_tokens: list[int] = (
            [int(SpecialToken.SOS_SEQ), int(SpecialToken.MAP_START)] + suffix if use_cfg else []
        )

        window_start_bin = int(round(window_start_ms / self.tokenizer_cfg.dt_bin_ms))
        generate_end_bin = self._context_bins + (self.tokenizer_cfg.generate_ms // self.tokenizer_cfg.dt_bin_ms)

        out_groups: list[tuple[int, list[Event]]] = []
        current_group: list[Event] = []
        current_raw_abs: int | None = None
        expecting_rel = False
        last_emitted_window_local: int | None = None
        min_spacing = max(0, self.sampling.min_abs_time_spacing_bins)

        cond_ids = torch.tensor(cond_tokens, dtype=torch.long, device=self.device).unsqueeze(0)
        cond_step, cond_cache = self.model.decode_step(
            cond_ids, memory=memory, cache=None, start_pos=0
        )
        cond_logits = cond_step[0, -1, : self._vocab_out]
        if use_cfg:
            uncond_ids = torch.tensor(uncond_tokens, dtype=torch.long, device=self.device).unsqueeze(0)
            uncond_step, uncond_cache = self.model.decode_step(
                uncond_ids, memory=memory, cache=None, start_pos=0
            )
            uncond_logits = uncond_step[0, -1, : self._vocab_out]
            logits = uncond_logits + guidance * (cond_logits - uncond_logits)
        else:
            uncond_cache = None
            logits = cond_logits

        cond_len = len(cond_tokens)
        uncond_len = len(uncond_tokens)

        while cond_len < self.max_decoder_len:
            if expecting_rel:
                assert current_raw_abs is not None
                rel = self._rel(last_raw_bin, current_raw_abs)
                next_id = self.vocab.encode_event(Event(EventType.REL_TIME, rel))
                last_raw_bin = current_raw_abs
                expecting_rel = False
            else:
                if self.sampling.disable_grammar:
                    masked_logits = logits.clone()
                else:
                    mask = grammar.current_mask().to(logits.device)
                    masked_logits = logits.masked_fill(~mask, float("-inf"))
                if self._bias_vector.abs().sum().item() > 0:
                    masked_logits = masked_logits + self._bias_vector.to(masked_logits.device)
                if min_spacing > 0 and last_emitted_window_local is not None:
                    lo = self._abs_start + last_emitted_window_local
                    hi = min(self._abs_end, self._abs_start + last_emitted_window_local + min_spacing)
                    masked_logits[lo:hi] = float("-inf")
                if self.sampling.eos_bias != 0.0:
                    eos_id = int(SpecialToken.EOS)
                    if 0 <= eos_id < self._vocab_out:
                        masked_logits[eos_id] = masked_logits[eos_id] + self.sampling.eos_bias
                is_time = self._abs_start <= (masked_logits.argmax().item()) < self._abs_end
                next_id = sample_next_token(masked_logits, self.sampling, is_time_token=is_time)
                if not self.sampling.disable_grammar:
                    grammar.update(next_id)
                if next_id == int(SpecialToken.EOS):
                    break
                decoded = self.vocab.decode_token(next_id)
                if isinstance(decoded, Event):
                    if decoded.type == EventType.ABS_TIME:
                        if current_group and current_raw_abs is not None:
                            out_groups.append((current_raw_abs, current_group))
                        window_local = decoded.value
                        if window_local >= generate_end_bin:
                            break
                        current_raw_abs = window_start_bin + window_local
                        current_group = []
                        expecting_rel = True
                        last_emitted_window_local = window_local
                    else:
                        current_group.append(decoded)

            step_in = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
            cond_step, cond_cache = self.model.decode_step(
                step_in, memory=memory, cache=cond_cache, start_pos=cond_len
            )
            cond_len += 1
            cond_logits = cond_step[0, -1, : self._vocab_out]
            if use_cfg:
                uncond_step, uncond_cache = self.model.decode_step(
                    step_in, memory=memory, cache=uncond_cache, start_pos=uncond_len
                )
                uncond_len += 1
                uncond_logits = uncond_step[0, -1, : self._vocab_out]
                logits = uncond_logits + guidance * (cond_logits - uncond_logits)
            else:
                logits = cond_logits

        if current_group and current_raw_abs is not None:
            out_groups.append((current_raw_abs, current_group))
        return _drop_trailing_open_objects(out_groups)

    def _rel(self, last: int | None, current: int) -> int:
        if last is None:
            return 0
        delta = current - last
        if delta < 0 or delta > self._max_rel_bin:
            return self._max_rel_bin
        return delta

    def _slice_mel(self, mel: np.ndarray, window_start_ms: float) -> np.ndarray:
        start_frame = int(round(window_start_ms / self.audio_cfg.hop_ms))
        end_frame = start_frame + self._frames_per_window
        window = mel[start_frame:end_frame]
        if window.shape[0] < self._frames_per_window:
            pad = np.zeros((self._frames_per_window - window.shape[0], window.shape[1]), dtype=window.dtype)
            window = np.concatenate([window, pad], axis=0)
        return window


_CLOSE_MARKERS: frozenset[EventType] = frozenset(
    {EventType.CIRCLE, EventType.SLIDER_END, EventType.SPINNER_END}
)


def _drop_trailing_open_objects(
    out_groups: list[tuple[int, list[Event]]],
) -> list[tuple[int, list[Event]]]:
    last_safe = -1
    for idx, (_, group) in enumerate(out_groups):
        if any(ev.type in _CLOSE_MARKERS for ev in group):
            last_safe = idx
    return out_groups[: last_safe + 1]
