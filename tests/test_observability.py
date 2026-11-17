"""Stage lifecycle, console capture, child failures and unchanged RNG streams."""
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import io
import json
import logging
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from ep_loop.huginn import tiny_huginn
from ep_loop.inference import generate
from ep_loop.observability import RunLogger, run_command
from scripts.watch_run_timeline import EventTail
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


class ObservabilityTests(unittest.TestCase):
    def logger(self, folder):
        return RunLogger(Path(folder)/'events.jsonl', Path(folder)/'console.log', 'test')

    def rows(self, folder):
        return [json.loads(line) for line in (Path(folder)/'events.jsonl').read_text().splitlines()]

    def test_nested_failure_has_balanced_times_traceback_and_console(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            original_stdout, original_stderr = sys.stdout, sys.stderr
            handler = logging.StreamHandler(sys.stderr)
            named = logging.getLogger('ep_loop_test_capture')
            named.addHandler(handler)
            try:
                with self.assertRaisesRegex(ValueError, 'bad response'):
                    with self.logger(folder) as observer:
                        print('stdout recorded')
                        print('stderr recorded', file=sys.stderr)
                        named.warning('preexisting logging handler recorded')
                        with observer.stage('outer'):
                            observer.start('inner')
                            raise ValueError('bad response')
                self.assertIs(sys.stdout, original_stdout)
                self.assertIs(sys.stderr, original_stderr)
                self.assertIs(handler.stream, original_stderr)
                rows = self.rows(folder)
                self.assertEqual(rows[-1]['status'], 'failed')
                self.assertIn('ValueError: bad response', rows[-1]['traceback'])
                starts = [r for r in rows if r['event']=='stage_start']
                ends = [r for r in rows if r['event']=='stage_end']
                self.assertEqual({r['stage_id'] for r in starts}, {r['stage_id'] for r in ends})
                self.assertTrue(all(r['status']=='failed' and r['seconds']>=0 for r in ends))
                for row in rows:
                    self.assertEqual(datetime.fromisoformat(row['timestamp']).utcoffset().total_seconds(), 0)
                    self.assertIn('session_id', row)
                console = (Path(folder)/'console.log').read_text()
                for text in ['stdout recorded', 'stderr recorded', 'preexisting logging handler recorded', 'ValueError: bad response']:
                    self.assertIn(text, console)
            finally:
                named.removeHandler(handler)

    def test_clean_system_exit_and_resume_append_separate_sessions(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stderr(io.StringIO()):
            for _ in range(2):
                with self.assertRaises(SystemExit):
                    with self.logger(folder):
                        raise SystemExit(0)
            rows = self.rows(folder)
            self.assertEqual(len({r['session_id'] for r in rows}), 2)
            self.assertTrue(all(r['status']=='completed' for r in rows if r['event']=='run_end'))

    def test_absolute_stage_times_context_and_rank_one_timeline(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stderr(io.StringIO()):
            with RunLogger(Path(folder)/'events.jsonl',Path(folder)/'console.log','test',rank=1) as observer:
                with observer.stage('response',update=7,id='question',sample=2):
                    with observer.stage('backward'):
                        pass
            rows=self.rows(folder)
            for row in rows:
                self.assertGreater(row['unix_time_ns'],0)
                self.assertGreater(row['monotonic_ns'],0)
                for key in ('started_at','finished_at'):
                    if key in row:
                        self.assertEqual(datetime.fromisoformat(row[key]),datetime.fromisoformat(row[key+'_local']))
                        self.assertTrue(row[key+'_local'].endswith('+08:00'))
            backward=next(r for r in rows if r.get('stage')=='backward' and r['event']=='stage_start')
            self.assertEqual((backward['update'],backward['id'],backward['sample']),(7,'question',2))
            timeline=(Path(folder)/'events.timeline.log').read_text()
            self.assertIn('stage_start: backward',timeline)
            self.assertIn('stage_end: backward',timeline)
            self.assertIn('rank=1',timeline)

    def test_heartbeat_records_generation_progress_and_thread_stops(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stderr(io.StringIO()), patch.dict(os.environ,{'EP_LOOP_HEARTBEAT_SECONDS':'0.01'}):
            with self.logger(folder) as observer:
                with observer.stage('generation',id='question',max_new_tokens=16384):
                    observer.update_generation_progress(response_tokens=1234,last_exit_depth=2)
                    observer.heartbeat()
                    threading.Event().wait(.035)
            self.assertFalse(observer.heartbeat_thread.is_alive())
            heartbeats=[r for r in self.rows(folder) if r['event']=='heartbeat' and r['active_stages']]
            self.assertGreaterEqual(len(heartbeats),2)
            progress=heartbeats[0]['active_stages'][-1]['progress']
            self.assertEqual(progress['response_tokens'],1234)
            self.assertEqual(progress['last_exit_depth'],2)

    def test_live_observer_reads_partial_lines_without_altering_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'events.jsonl'
            row=dict(event='stage_start',stage='decode',stage_id=1,rank=0,pid=os.getpid(),
                     timestamp='2026-11-16T08:00:00+00:00',started_at='2026-11-16T08:00:00+00:00')
            encoded=(json.dumps(row)+'\n').encode()
            path.write_bytes(encoded[:20])
            tail=EventTail(path)
            self.assertEqual(list(tail.read()),[])
            with path.open('ab') as stream:stream.write(encoded[20:])
            self.assertEqual(list(tail.read()),[row])
            self.assertEqual(list(tail.read()),[])
            snapshot=tail.snapshot(ZoneInfo('Asia/Shanghai'))
            self.assertEqual(snapshot['active_stages'][0]['started_at_local'],'2026-11-16T16:00:00.000+08:00')
            self.assertEqual(path.read_bytes(),encoded)

    def test_live_observer_emits_new_nodes_and_exits_with_launcher(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            source=root/'launcher.events.jsonl'
            stamp=datetime.now(timezone.utc).isoformat()
            start=dict(event='stage_start',stage='generation',stage_id=1,rank=0,pid=os.getpid(),
                       timestamp=stamp,started_at=stamp)
            source.write_text(json.dumps(start)+'\n')
            with subprocess.Popen([sys.executable,'-c','import time;time.sleep(2)']) as launcher:
                command=[sys.executable,str(ROOT/'scripts/watch_run_timeline.py'),'--run',str(root),
                         '--launcher-pid',str(launcher.pid),'--poll-seconds','.02','--heartbeat-seconds','.05']
                with subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE) as watcher:
                    output=root/'timeline.live.jsonl'
                    deadline=time.monotonic()+5
                    while (not output.exists() or 'watch_heartbeat' not in output.read_text()) and time.monotonic()<deadline:
                        threading.Event().wait(.01)
                    self.assertTrue(output.exists())
                    end=start | dict(event='stage_end',finished_at=stamp,status='completed',seconds=.1)
                    with source.open('a') as stream:stream.write(json.dumps(end)+'\n')
                    stdout,stderr=watcher.communicate(timeout=6)
                    self.assertEqual(watcher.returncode,0,stderr.decode())
            rows=[json.loads(line) for line in output.read_text().splitlines()]
            observed=next(row for row in rows if row['event']=='source_event')
            self.assertEqual(observed['source_event'],'stage_end')
            self.assertEqual(rows[-1]['reason'],'launcher_exited')
            self.assertEqual(source.read_text(),json.dumps(start)+'\n'+json.dumps(end)+'\n')

    def test_subprocess_combined_raw_log_and_failure_exit_code(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            path = Path(folder)/'child.log'
            with self.logger(folder) as observer:
                command = [sys.executable, '-u', '-c', "import sys; print('child stdout'); print('child stderr', file=sys.stderr); sys.exit(7)"]
                with self.assertRaises(subprocess.CalledProcessError) as raised:
                    run_command(command, os.environ.copy(), ROOT, observer, 'child', path)
                self.assertEqual(raised.exception.returncode, 7)
            self.assertEqual(path.read_text(), 'child stdout\nchild stderr\n')
            end = next(r for r in self.rows(folder) if r['event']=='stage_end')
            self.assertEqual(end['returncode'], 7)
            self.assertEqual(end['status'], 'failed')

    def test_rank_one_disables_bars_without_disabling_events(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stderr(io.StringIO()):
            with RunLogger(Path(folder)/'events.jsonl', Path(folder)/'console.log', 'test', rank=1) as observer:
                self.assertFalse(observer.show_progress)
                self.assertFalse(observer.show_tokens)
                for _ in observer.progress(range(3), desc='hidden bar'):
                    pass
            self.assertNotIn('hidden bar', (Path(folder)/'console.log').read_text())
            self.assertTrue(all(r['rank']==1 for r in self.rows(folder)))

    def test_generation_logging_keeps_samples_and_global_rng_identical(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        backend = tiny_huginn(ROOT/'models/huginn-0125')
        backend.enable_student_exit()
        torch.manual_seed(42)
        first = generate(backend, [1,20,30], 4, 6, 1., 1., torch.Generator().manual_seed(42), 'learned', .5, True)
        expected_rng = torch.get_rng_state()
        with tempfile.TemporaryDirectory() as folder, redirect_stderr(io.StringIO()), patch.dict(os.environ, {'EP_LOOP_PROGRESS':'1','EP_LOOP_TOKEN_PROGRESS':'1'}):
            torch.manual_seed(42)
            python_state = random.getstate()
            with self.logger(folder) as observer:
                second = generate(backend, [1,20,30], 4, 6, 1., 1., torch.Generator().manual_seed(42), 'learned', .5, True,
                                  observer=observer, progress=True, progress_desc='Sample tokens')
            self.assertEqual(first.token_ids, second.token_ids)
            self.assertEqual(first.exit_depths, second.exit_depths)
            self.assertEqual(python_state, random.getstate())
            torch.testing.assert_close(expected_rng, torch.get_rng_state(), rtol=0, atol=0)
            ends = [r for r in self.rows(folder) if r['event']=='stage_end']
            expected_stages = {'generation','prefill'} | ({'decode'} if len(first.token_ids)>1 else set())
            self.assertEqual({r['stage'] for r in ends}, expected_stages)
            generation = next(r for r in ends if r['stage']=='generation')
            self.assertEqual(generation['response_tokens'], len(first.token_ids))
            self.assertIn('Sample tokens', (Path(folder)/'console.log').read_text())
            self.assertIn(f'{len(first.token_ids)}/6', (Path(folder)/'console.log').read_text())
            self.assertIn('+08:00] [rank=0]',(Path(folder)/'console.log').read_text())


if __name__=='__main__':
    unittest.main()
