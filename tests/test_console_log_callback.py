import sys
from types import SimpleNamespace

from skp.callbacks.console_log import ConsoleLogCallback


def test_console_log_callback_mirrors_rank_zero_output(tmp_path, capsys):
    log_path = tmp_path / "fold0" / "training.log"
    callback = ConsoleLogCallback(str(log_path))
    trainer = SimpleNamespace(is_global_zero=True)

    callback.on_fit_start(trainer, None)
    print("validation metric: 0.8123")
    print("warning from stderr", file=sys.stderr)
    callback.on_fit_end(trainer, None)

    captured = capsys.readouterr()
    contents = log_path.read_text()
    assert "validation metric: 0.8123" in captured.out
    assert "warning from stderr" in captured.err
    assert "validation metric: 0.8123" in contents
    assert "warning from stderr" in contents
    assert "training started" in contents
    assert "training finished" in contents


def test_console_log_callback_does_nothing_on_nonzero_rank(tmp_path):
    log_path = tmp_path / "fold1" / "training.log"
    callback = ConsoleLogCallback(str(log_path))

    callback.on_fit_start(SimpleNamespace(is_global_zero=False), None)
    callback.on_fit_end(None, None)

    assert not log_path.exists()
