class General:
  def __init__(
    self, *, 
    raw: str,
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

    if(raw):
      kv = self.key_value(raw)
      for k, v in kv.items():
        if hasattr(self, k.lower()):
          setattr(self, k.lower(), type(getattr(self, k.lower()))(v))

  def key_value(self, raw: str) -> dict[str, str]:
    rows = raw.splitlines()
    [key, value] = [row.split(":", 1) for row in rows]
    return dict(zip(key, value))
