from marble_aim.__main__ import build_parser, default_config_path


def test_default_config_uses_each_users_local_app_data(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

    assert default_config_path() == tmp_path / "MarbleAim" / "config.json"


def test_config_argument_remains_available_for_portable_debugging(tmp_path):
    path = tmp_path / "portable.json"

    args = build_parser().parse_args(["--config", str(path)])

    assert args.config == path
