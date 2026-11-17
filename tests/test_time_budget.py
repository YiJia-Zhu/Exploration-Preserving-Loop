"""Bounded experiment cleanup and complete-checkpoint recovery markers."""
from contextlib import redirect_stdout,redirect_stderr
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from ep_loop.process_control import process_record,terminate_experiment

ROOT=Path(__file__).resolve().parents[1]


class TimeBudgetTests(unittest.TestCase):
    def test_cleanup_only_signals_owned_process_tree(self):
        code="import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']); print(p.pid,flush=True); time.sleep(30)"
        with subprocess.Popen([sys.executable,'-c',code],stdout=subprocess.PIPE,text=True) as root,\
                subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']) as unrelated:
            try:
                child=int(root.stdout.readline())
                rows=terminate_experiment(root.pid,grace_seconds=.2)
                self.assertIn(root.pid,{row['pid'] for row in rows})
                self.assertIn(child,{row['pid'] for row in rows})
                self.assertNotIn(unrelated.pid,{row['pid'] for row in rows})
                root.wait(timeout=3)
                self.assertIsNone(unrelated.poll())
                state=process_record(child)
                self.assertTrue(state is None or state['state']=='Z')
            finally:
                if root.poll() is None:root.kill()
                unrelated.terminate();unrelated.wait(timeout=3)

    def test_watchdog_stops_before_budget_and_lists_complete_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            panel=Path(directory)/'panel';panel.mkdir()
            checkpoint=Path(directory)/'case_endpoint/ours/checkpoints/update_000001'
            checkpoint.mkdir(parents=True)
            (checkpoint/'checkpoint.complete.json').write_text('{"resumable":true}\n')
            with subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']) as root:
                command=[sys.executable,str(ROOT/'scripts/watch_run_budget.py'),'--pid',str(root.pid),
                         '--hours',str(1/3600),'--started-monotonic',str(time.monotonic()),
                         '--panel',str(panel),'--output-root',directory,'--run-prefix','case']
                subprocess.run(command,check=True,timeout=5)
                root.wait(timeout=3)
            report=json.loads((panel/'time_budget.json').read_text())
            self.assertEqual(report['state'],'stopped_at_time_budget')
            self.assertEqual(report['complete_checkpoints'],[str(checkpoint)])

    def test_checkpoint_marker_appears_after_full_recovery_state(self):
        from ep_loop.training import train
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
            train(ROOT/'configs/default.json',directory,'cpu',True,stop_after_updates=1)
            folder=Path(directory)/'checkpoints/update_000001'
            marker=json.loads((folder/'checkpoint.complete.json').read_text())
            self.assertTrue(marker['resumable'])
            self.assertEqual(marker['completed_updates'],1)
            for name in ['model.safetensors','training_state.pt','rng_rank0.pt']:
                self.assertTrue((folder/name).is_file(),name)


if __name__=='__main__':unittest.main()
