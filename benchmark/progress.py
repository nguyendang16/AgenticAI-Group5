from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field


@dataclass
class ProgressReporter:
    """Lightweight terminal progress for long benchmark phases (no extra deps)."""

    label: str
    total: int
    current: int = 0
    _started_at: float = field(default_factory=time.monotonic)

    def start(self, message: str = '') -> None:
        self._print_line(0, message or 'starting')

    def advance(self, message: str = '') -> None:
        self.current = min(self.total, self.current + 1)
        self._print_line(self.current, message)

    def skip(self, message: str = '') -> None:
        self._print_line(self.current, f'SKIP {message}'.strip())

    def finish(self, message: str = '') -> None:
        elapsed = time.monotonic() - self._started_at
        suffix = f' — {message}' if message else ''
        print(
            f'\n[{self.label}] done ({self.current}/{self.total}) in {elapsed:.1f}s{suffix}',
            flush=True,
        )

    def _print_line(self, step: int, message: str) -> None:
        width = max(len(str(self.total)), 2)
        bar_width = 24
        filled = 0 if self.total <= 0 else int(bar_width * step / self.total)
        bar = '#' * filled + '-' * (bar_width - filled)
        msg = f' {message}' if message else ''
        print(
            f'[{self.label}] [{step:>{width}}/{self.total}] |{bar}|{msg}',
            flush=True,
        )


def phase_banner(phase: str, *, detail: str = '') -> None:
    line = '=' * 72
    print(f'\n{line}', flush=True)
    print(f'  BENCHMARK PHASE: {phase.upper()}', flush=True)
    if detail:
        print(f'  {detail}', flush=True)
    print(f'{line}\n', flush=True)


def warn_if_live(command: str) -> None:
    if command == 'run':
        print(
            'NOTE: benchmark run uses OpenAI for reviews. '
            'Ensure no other review workers are running.',
            flush=True,
        )


def pipeline_exit(rc: int, step_name: str) -> int:
    if rc != 0:
        print(f'\nPIPELINE STOPPED at step "{step_name}" (exit {rc})', file=sys.stderr, flush=True)
    return rc
