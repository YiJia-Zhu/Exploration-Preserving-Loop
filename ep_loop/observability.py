"""Rank-local stage events, console capture and throttled progress bars.

No Torch import, CUDA synchronization, random-stream consumption, token-level
file writes or model inspection. Timings are host wall time, not GPU profiles.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import itertools
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
import uuid
import weakref
from zoneinfo import ZoneInfo

from tqdm.auto import tqdm


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class TimestampedTqdm(tqdm):
    """Stamp actual display refreshes, preserving tqdm's throttling."""
    def __init__(self, *args, log_timezone, log_rank, **kwargs):
        self.log_timezone, self.log_rank = log_timezone, log_rank
        super().__init__(*args, **kwargs)

    def display(self, msg=None, pos=None):
        if msg != '':
            stamp = datetime.now(self.log_timezone).isoformat(timespec='milliseconds')
            msg = f'[{stamp}] [rank={self.log_rank}] {str(self) if msg is None else msg}'
        return super().display(msg, pos)


def env_bool(name, default=True):
    value = os.environ.get(name, str(int(default))).lower()
    if value not in {"0", "1", "true", "false"}:
        raise ValueError(f"{name} must be 0/1 or false/true")
    return value in {"1", "true"}


@contextmanager
def model_progress(enabled):
    """Also suppress Transformers checkpoint bars on non-display ranks."""
    from transformers.utils import logging as hf_logging
    previous = hf_logging.is_progress_bar_enabled()
    if not enabled:
        hf_logging.disable_progress_bar()
    try:
        yield
    finally:
        if previous:
            hf_logging.enable_progress_bar()
        else:
            hf_logging.disable_progress_bar()


class Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, text):
        self.log.write(text)
        return self.stream.write(text)

    def flush(self):
        self.log.flush()
        self.stream.flush()

    def __getattr__(self, key):
        return getattr(self.stream, key)


class RunLogger:
    def __init__(self, events_path, console_path, name, rank=0):
        self.events_path, self.console_path = Path(events_path), Path(console_path)
        self.timeline_path = self.events_path.with_suffix('.timeline.log')
        self.name, self.rank = name, rank
        self.session_id = uuid.uuid4().hex
        self.started_at = utc_now()
        self.clock = time.perf_counter()
        self.timezone = ZoneInfo(os.environ.get("EP_LOOP_LOG_TIMEZONE", "Asia/Shanghai"))
        self.show_progress = env_bool("EP_LOOP_PROGRESS") and rank == 0
        self.show_tokens = self.show_progress and env_bool("EP_LOOP_TOKEN_PROGRESS")
        self.mininterval = float(os.environ.get("EP_LOOP_PROGRESS_MIN_INTERVAL", "1.0"))
        if not 0 < self.mininterval < float("inf"):
            raise ValueError("EP_LOOP_PROGRESS_MIN_INTERVAL must be finite and positive")
        self.stage_timestamps = env_bool('EP_LOOP_STAGE_TIMESTAMPS')
        self.heartbeat_seconds = float(os.environ.get('EP_LOOP_HEARTBEAT_SECONDS', '30'))
        if not 0 <= self.heartbeat_seconds < float('inf'):
            raise ValueError('EP_LOOP_HEARTBEAT_SECONDS must be finite and nonnegative')
        self.lock, self.stop_heartbeat = threading.RLock(), threading.Event()
        self.heartbeat_thread = None
        self.stack, self.bars = [], weakref.WeakSet()
        self.counter = itertools.count(1)

    def __enter__(self):
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self.console_path.parent.mkdir(parents=True, exist_ok=True)
        self.events = self.events_path.open("a", buffering=1)
        self.console = self.console_path.open("a", buffering=1)
        self.timeline = self.timeline_path.open('a', buffering=1)
        self.stdout, self.stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(self.stdout, self.console), Tee(self.stderr, self.console)
        self.tee_stdout, self.tee_stderr = sys.stdout, sys.stderr
        for handler in self.stream_handlers():
            if handler.stream is self.stdout:
                handler.setStream(self.tee_stdout)
            elif handler.stream is self.stderr:
                handler.setStream(self.tee_stderr)
        self.event("run_start", echo=True, progress=self.show_progress, token_progress=self.show_tokens)
        if self.heartbeat_seconds:
            self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True,
                                                     name=f'ep-loop-log-heartbeat-{self.rank}')
            self.heartbeat_thread.start()
        return self

    @staticmethod
    def stream_handlers():
        loggers = [logging.getLogger(), *logging.Logger.manager.loggerDict.values()]
        seen = set()
        for logger in loggers:
            if isinstance(logger, logging.Logger):
                for handler in logger.handlers:
                    if isinstance(handler, logging.StreamHandler) and id(handler) not in seen:
                        seen.add(id(handler))
                        stream = getattr(type(handler), "stream", None)
                        # Some libraries expose a read-only stream property
                        # that follows sys.stderr dynamically. It needs no swap.
                        if isinstance(stream, property) and stream.fset is None:
                            continue
                        yield handler

    def event(self, event, echo=False, **fields):
        with self.lock:
            unix_ns=time.time_ns()
            now=datetime.fromtimestamp(unix_ns/1e9,timezone.utc)
            row = dict(event=event, timestamp=now.isoformat(timespec="milliseconds"),
                local_time=now.astimezone(self.timezone).isoformat(timespec="milliseconds"),
                session_id=self.session_id, name=self.name, rank=self.rank, pid=os.getpid(),
                parent_session_id=os.environ.get("EP_LOOP_PARENT_SESSION_ID"),
                elapsed_seconds=time.perf_counter()-self.clock, **fields)
            row.update(unix_time_ns=unix_ns, monotonic_ns=time.monotonic_ns())
            for key in ('started_at', 'finished_at'):
                if row.get(key):
                    row[key+'_local'] = datetime.fromisoformat(row[key]).astimezone(self.timezone).isoformat(timespec='milliseconds')
            self.events.write(json.dumps(row, ensure_ascii=False)+"\n")
            if event in {'run_start', 'run_end', 'stage_start', 'stage_end', 'heartbeat'}:
                label = fields.get('stage', self.name)
                details = ' '.join(f'{key}={fields[key]}' for key in
                    ('update','id','sample','status','response_tokens','seconds','started_at_local','finished_at_local')
                    if key in fields)
                for key in ('started_at_local','finished_at_local'):
                    if key in row and key not in fields:
                        details += f' {key}={row[key]}'
                if event == 'heartbeat':
                    details += ' active=' + json.dumps(fields.get('active_stages',[]),ensure_ascii=False,separators=(',',':'))
                message=f"[{row['local_time']}] [rank={self.rank} pid={row['pid']}] {event}: {label} {details}".rstrip()
                self.timeline.write(message+'\n')
                if echo and self.rank == 0:
                    tqdm.write(message, file=sys.stderr)
        return row

    def start(self, stage, echo=None, **fields):
        with self.lock:
            context={key:value for span in self.stack for key,value in span['fields'].items()
                     if key in {'update','id','sample','policy','max_new_tokens'}}
            fields=context | fields
            span = dict(stage=stage, stage_id=next(self.counter),
                        parent_stage_id=self.stack[-1]["stage_id"] if self.stack else None,
                        started_at=utc_now(), fields=fields,
                        echo=self.stage_timestamps if echo is None else echo, clock=time.perf_counter())
            self.stack.append(span)
            self.event("stage_start", echo=span['echo'], **{k:v for k,v in span.items() if k not in {"clock", "echo", "fields"}}, **fields)
        return span

    def finish(self, span, status="completed", **fields):
        with self.lock:
            if not self.stack or self.stack[-1] is not span:
                raise RuntimeError("Stage events must close in nesting order")
            self.stack.pop()
            return self.event("stage_end", echo=span["echo"],
                stage=span["stage"], stage_id=span["stage_id"], parent_stage_id=span["parent_stage_id"],
                started_at=span["started_at"], finished_at=utc_now(),
                seconds=time.perf_counter()-span["clock"], status=status, **(span["fields"] | fields))

    def update_generation_progress(self, **fields):
        """Host integers only; defer file I/O to the periodic heartbeat."""
        with self.lock:
            for span in reversed(self.stack):
                if span['stage'] == 'generation':
                    span['fields']['progress'] = fields
                    break

    def heartbeat(self):
        with self.lock:
            now=time.perf_counter()
            active=[dict(stage=span['stage'],stage_id=span['stage_id'],
                         started_at=span['started_at'],
                         started_at_local=datetime.fromisoformat(span['started_at']).astimezone(self.timezone).isoformat(timespec='milliseconds'),
                         seconds=now-span['clock'],**span['fields']) for span in self.stack]
            return self.event('heartbeat',echo=False,active_stages=active)

    def _heartbeat_loop(self):
        while not self.stop_heartbeat.wait(self.heartbeat_seconds):
            self.heartbeat()

    @contextmanager
    def stage(self, stage, echo=None, **fields):
        span = self.start(stage, echo, **fields)
        result = {}
        try:
            yield result
        except BaseException as exc:
            while self.stack and self.stack[-1] is not span:
                self.finish(self.stack[-1], "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                            error_type=type(exc).__name__, error=str(exc))
            self.finish(span, "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        **(result | dict(error_type=type(exc).__name__, error=str(exc))))
            raise
        else:
            self.finish(span, **result)

    def progress(self, iterable=None, *, tokens=False, **kwargs):
        bar = TimestampedTqdm(iterable, disable=not (self.show_tokens if tokens else self.show_progress),
                   mininterval=self.mininterval, dynamic_ncols=True, file=sys.stderr,
                   log_timezone=self.timezone,log_rank=self.rank,**kwargs)
        self.bars.add(bar)
        return bar

    def metadata(self):
        return dict(session_id=self.session_id, started_at=self.started_at,
            started_at_local=datetime.fromisoformat(self.started_at).astimezone(self.timezone).isoformat(timespec='milliseconds'),
            events=str(self.events_path.resolve()), console=str(self.console_path.resolve()),
            timeline=str(self.timeline_path.resolve()),heartbeat_seconds=self.heartbeat_seconds,
            stage_timestamps=self.stage_timestamps,
            timezone=str(self.timezone), progress=self.show_progress, token_progress=self.show_tokens,
            progress_min_interval=self.mininterval, timing="host wall time; no additional CUDA synchronization")

    def __exit__(self, kind, exc, tb):
        self.stop_heartbeat.set()
        if self.heartbeat_thread is not None:
            self.heartbeat_thread.join()
        success = exc is None or isinstance(exc, SystemExit) and exc.code in (None, 0)
        status = "completed" if success else "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        try:
            for bar in sorted(self.bars, key=lambda bar:abs(bar.pos), reverse=True):
                bar.close()
            while self.stack:
                self.finish(self.stack[-1], status, error=str(exc) if exc else "unclosed stage")
            fields = dict(status=status, started_at=self.started_at, finished_at=utc_now(),
                          seconds=time.perf_counter()-self.clock)
            if not success:
                fields.update(error_type=kind.__name__, error=str(exc),
                              traceback="".join(traceback.format_exception(kind, exc, tb)))
                self.console.write(fields["traceback"])
            self.event("run_end", echo=True, **fields)
        finally:
            try:
                for handler in self.stream_handlers():
                    if handler.stream is self.tee_stdout:
                        handler.setStream(self.stdout)
                    elif handler.stream is self.tee_stderr:
                        handler.setStream(self.stderr)
            finally:
                sys.stdout, sys.stderr = self.stdout, self.stderr
                self.events.close()
                self.console.close()
                self.timeline.close()


def run_command(cmd, env, cwd, logger, label, log_path):
    """Stream child stdout/stderr to the terminal and a persistent raw log."""
    import codecs
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with logger.stage("subprocess", echo=True, label=label, command=list(map(str, cmd)),
                      log_path=str(path), cuda_visible_devices=env.get("CUDA_VISIBLE_DEVICES")) as result:
        with path.open("ab") as log:
            child_env = {**env, "EP_LOOP_PARENT_SESSION_ID": logger.session_id}
            with subprocess.Popen(cmd, env=child_env, cwd=cwd, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, bufsize=0) as process:
                decoder = codecs.getincrementaldecoder("utf-8")("replace")
                try:
                    while chunk := process.stdout.read(65536):
                        log.write(chunk)
                        log.flush()
                        sys.stdout.write(decoder.decode(chunk))
                        sys.stdout.flush()
                    sys.stdout.write(decoder.decode(b"", final=True))
                    code = process.wait()
                except BaseException:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise
        result["returncode"] = code
        if code:
            raise subprocess.CalledProcessError(code, cmd)
