import click
from click.testing import CliRunner

from src.config import AppConfig, load_config, with_config

_CFG = "config/config.yaml"
_ROOT = "config"


class TestLoader:
    def test_default_model_version(self):
        assert load_config(_CFG).model.version == "md"

    def test_audio_version(self):
        assert load_config(_CFG).audio.version == "v1"

    def test_tokenizer_version(self):
        assert load_config(_CFG).tokenizer.version == "v1"

    def test_dataset_version(self):
        assert load_config(_CFG).dataset.version == "v1"

    def test_returns_app_config(self):
        assert isinstance(load_config(_CFG), AppConfig)

    def test_override_model_sm(self):
        cfg = load_config(_CFG, section_overrides={"model": f"{_ROOT}/models/sm.yaml"})
        assert cfg.model.version == "sm"

    def test_override_model_lg(self):
        cfg = load_config(_CFG, section_overrides={"model": f"{_ROOT}/models/lg.yaml"})
        assert cfg.model.version == "lg"

    def test_override_model_xl(self):
        cfg = load_config(_CFG, section_overrides={"model": f"{_ROOT}/models/xl.yaml"})
        assert cfg.model.version == "xl"

    def test_dotlist_override(self):
        cfg = load_config(_CFG, dotlist=["model.encoder.d_model=256"])
        assert cfg.model.encoder.d_model == 256

    def test_nested_dot_access(self):
        cfg = load_config(_CFG)
        assert cfg.training.lr_scheduler.type == "cosine"
        assert cfg.model.encoder.d_model == 512


class TestWithConfig:
    def _version_cmd(self) -> click.Command:
        @click.command()
        @with_config
        def cmd(cfg: AppConfig) -> None:
            click.echo(cfg.model.version)

        return cmd

    def test_default_loads(self):
        result = CliRunner().invoke(self._version_cmd(), [])
        assert result.exit_code == 0
        assert "md" in result.output

    def test_config_model_sm(self):
        result = CliRunner().invoke(self._version_cmd(), ["--config-model=sm"])
        assert result.exit_code == 0
        assert "sm" in result.output

    def test_config_model_lg(self):
        result = CliRunner().invoke(self._version_cmd(), ["--config-model=lg"])
        assert result.exit_code == 0
        assert "lg" in result.output

    def test_config_model_xl(self):
        result = CliRunner().invoke(self._version_cmd(), ["--config-model=xl"])
        assert result.exit_code == 0
        assert "xl" in result.output

    def test_config_audio(self):
        @click.command()
        @with_config
        def cmd(cfg: AppConfig) -> None:
            click.echo(cfg.audio.version)

        result = CliRunner().invoke(cmd, ["--config-audio=v1"])
        assert result.exit_code == 0
        assert "v1" in result.output

    def test_set_override(self):
        @click.command()
        @with_config
        def cmd(cfg: AppConfig) -> None:
            click.echo(cfg.model.encoder.d_model)

        result = CliRunner().invoke(cmd, ["--set", "model.encoder.d_model=256"])
        assert result.exit_code == 0
        assert "256" in result.output

    def test_extra_click_options_preserved(self):
        @click.command()
        @click.option("--name", default="world")
        @with_config
        def cmd(name: str, cfg: AppConfig) -> None:
            click.echo(f"{name}-{cfg.model.version}")

        result = CliRunner().invoke(cmd, ["--name=osu"])
        assert result.exit_code == 0
        assert "osu-md" in result.output
