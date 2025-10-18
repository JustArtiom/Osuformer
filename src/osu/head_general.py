import re

class General:
  def __init__(
    self, *, 
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
    samples_match_playback_rate: int = 0
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

    if raw:
      kv = self.key_value(raw)
      for k, v in kv.items():
        attr = self._normalize_key(k)
        if hasattr(self, attr):
          current_value = getattr(self, attr)
          try:
            setattr(self, attr, type(current_value)(v))
          except (TypeError, ValueError):
            setattr(self, attr, v)

  def key_value(self, raw: str) -> dict[str, str]:
    rows = raw.splitlines()
    key_value_pairs = []
    for row in rows:
      if ":" not in row:
        continue
      key, value = (segment.strip() for segment in row.split(":", 1))
      key_value_pairs.append((key, value))
    return dict(key_value_pairs)

  def _normalize_key(self, key: str) -> str:
    key = key.strip().replace(" ", "_")
    key = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", key)
    key = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key)
    return key.lower()

  def __str__(self) -> str:
    return f"AudioFilename: {self.audio_filename}\n" \
      f"AudioLeadIn: {self.audio_lead_in}\n" \
      f"PreviewTime: {self.preview_time}\n" \
      f"Countdown: {self.countdown}\n" \
      f"SampleSet: {self.sample_set}\n" \
      f"StackLeniency: {self.stack_leniency}\n" \
      f"Mode: {self.mode}\n" \
      f"LetterboxInBreaks: {self.letterbox_in_breaks}\n" \
      f"UseSkinSprites: {self.use_skin_sprites}\n" \
      f"OverlayPosition: {self.overlay_position}\n" \
      f"SkinPreference: {self.skin_preference}\n" \
      f"EpilepsyWarning: {self.epilepsy_warning}\n" \
      f"CountdownOffset: {self.countdown_offset}\n" \
      f"SpecialStyle: {self.special_style}\n" \
      f"WidescreenStoryboard: {self.widescreen_storyboard}\n" \
      f"SamplesMatchPlaybackRate: {self.samples_match_playback_rate}\n"
