import re

from ..utils import fmt


class General:
    def __init__(
        self,
        *,
        raw: str = "",
        audio_filename: str = "",
        audio_lead_in: int = 0,
        preview_time: int = -1,
        countdown: int = 0,
        sample_set: str = "Normal",
        stack_leniency: float = 0.7,
        mode: int = 0,
        letterbox_in_breaks: int = 0,
        use_skin_sprites: int = 0,
        overlay_position: str = "NoChange",
        skin_preference: str = "",
        epilepsy_warning: int = 0,
        countdown_offset: int = 0,
        special_style: int = 0,
        widescreen_storyboard: int = 0,
        samples_match_playback_rate: int = 0,
        sample_volume: int = 0,
    ):
        self.audio_filename = audio_filename
        self.audio_lead_in = audio_lead_in
        self.preview_time = preview_time
        self.countdown = countdown
        self.sample_set = sample_set
        self.stack_leniency = stack_leniency
        self.mode = mode
        self.letterbox_in_breaks = letterbox_in_breaks
        self.use_skin_sprites = use_skin_sprites
        self.overlay_position = overlay_position
        self.skin_preference = skin_preference
        self.epilepsy_warning = epilepsy_warning
        self.countdown_offset = countdown_offset
        self.special_style = special_style
        self.widescreen_storyboard = widescreen_storyboard
        self.samples_match_playback_rate = samples_match_playback_rate
        self.sample_volume = sample_volume

        if raw:
            kv = self._key_value(raw)
            for k, v in kv.items():
                attr = self._normalize_key(k)
                if hasattr(self, attr):
                    current_value = getattr(self, attr)
                    try:
                        setattr(self, attr, type(current_value)(v))
                    except (TypeError, ValueError):
                        setattr(self, attr, v)

    def _key_value(self, raw: str) -> dict[str, str]:
        result: dict[str, str] = {}
        for row in raw.splitlines():
            if ":" not in row:
                continue
            key, value = (s.strip() for s in row.split(":", 1))
            result[key] = value
        return result

    def _normalize_key(self, key: str) -> str:
        key = key.strip().replace(" ", "_")
        key = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", key)
        key = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key)
        return key.lower()

    def __str__(self) -> str:
        lines = [
            f"AudioFilename: {self.audio_filename}",
            f"AudioLeadIn: {self.audio_lead_in}",
            f"PreviewTime: {self.preview_time}",
            f"Countdown: {self.countdown}",
            f"SampleSet: {self.sample_set}",
            f"StackLeniency: {fmt(self.stack_leniency)}",
            f"Mode: {self.mode}",
        ]
        if self.letterbox_in_breaks:
            lines.append(f"LetterboxInBreaks: {self.letterbox_in_breaks}")
        if self.use_skin_sprites:
            lines.append(f"UseSkinSprites: {self.use_skin_sprites}")
        if self.overlay_position != "NoChange":
            lines.append(f"OverlayPosition: {self.overlay_position}")
        if self.skin_preference:
            lines.append(f"SkinPreference: {self.skin_preference}")
        if self.epilepsy_warning:
            lines.append(f"EpilepsyWarning: {self.epilepsy_warning}")
        if self.countdown_offset:
            lines.append(f"CountdownOffset: {self.countdown_offset}")
        if self.mode == 3 and self.special_style:
            lines.append(f"SpecialStyle: {self.special_style}")
        if self.widescreen_storyboard:
            lines.append(f"WidescreenStoryboard: {self.widescreen_storyboard}")
        if self.samples_match_playback_rate:
            lines.append(f"SamplesMatchPlaybackRate: {self.samples_match_playback_rate}")
        if self.sample_volume:
            lines.append(f"SampleVolume: {self.sample_volume}")
        return "\n".join(lines)
