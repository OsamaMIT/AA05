from __future__ import annotations

import pytest

from chrono_a2rl.cli import main


def test_aa_help_lists_main_commands(capsys) -> None:
    main(["--help"])
    captured = capsys.readouterr()

    assert "usage: aa <command> [options]" in captured.out
    assert "aa help [command]" in captured.out
    assert "train" in captured.out
    assert "eval" in captured.out
    assert "watch" in captured.out
    assert "optimize" in captured.out


def test_aa_help_subcommand_lists_main_commands(capsys) -> None:
    main(["help"])
    captured = capsys.readouterr()

    assert "usage: aa <command> [options]" in captured.out
    assert "aa help train" in captured.out


def test_aa_help_train_shows_train_options(capsys) -> None:
    main(["help", "train"])
    captured = capsys.readouterr()

    assert "usage: aa train" in captured.out
    assert "--total-timesteps" in captured.out


def test_aa_help_watch_shows_camera_options(capsys) -> None:
    main(["help", "watch"])
    captured = capsys.readouterr()

    assert "usage: aa watch" in captured.out
    assert "--camera" in captured.out
    assert "--zoom-radius" in captured.out


def test_aa_help_optimize_shows_profile_options(capsys) -> None:
    main(["help", "optimize"])
    captured = capsys.readouterr()

    assert "usage: aa optimize" in captured.out
    assert "--iterations" in captured.out
    assert "--backend" in captured.out


def test_aa_unknown_command_exits(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["unknown"])

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "Unknown command: unknown" in captured.err
