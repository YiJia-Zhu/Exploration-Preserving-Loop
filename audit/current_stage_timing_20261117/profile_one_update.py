from pathlib import Path
from datetime import datetime,timezone
import json,os,sys,time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from ep_loop.fastpath.actor import VllmActor
original=VllmActor.__init__
def timed_init(self,*args,**kwargs):
    start=time.perf_counter();original(self,*args,**kwargs)
    row=dict(event='engine_initialization',kind='teacher' if kwargs.get('teacher_trace') else 'student',seconds=time.perf_counter()-start,timestamp=datetime.now(timezone.utc).isoformat())
    with (Path(sys.argv[2])/('engine_initialization_rank'+os.environ.get('RANK','0')+'.jsonl')).open('a') as f:f.write(json.dumps(row)+'\n')
VllmActor.__init__=timed_init
from ep_loop.fastpath.training import train
train(sys.argv[1],sys.argv[2],'cuda',stop_after_updates=1)
