from benchmark.progress import ProgressReporter


def test_progress_reporter_advances(capsys):
    progress = ProgressReporter(label='test', total=2)
    progress.start('init')
    progress.advance('step-a')
    progress.advance('step-b')
    progress.finish('ok')
    captured = capsys.readouterr().out
    assert '[test]' in captured
    assert '1/2' in captured
    assert '2/2' in captured
    assert 'done' in captured
