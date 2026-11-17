"""Controlled correctness verifier integration: synthetic outputs, real tiny loss/DDP."""
import os
from pathlib import Path
import re
import sys
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from ep_loop.data import ByteTokenizer
from ep_loop.inference import GeneratedResponse
from ep_loop.training import train
mode,config,folder=sys.argv[1:4];resume=sys.argv[4] if len(sys.argv)>4 else None
rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));calls=[0];tokenizer=ByteTokenizer()
def generated(backend,prompt,*args,**kwargs):
 number=int(re.search(r'What is (\d+)\+1',tokenizer.decode(prompt))[1])+1
 index=calls[0]%4;calls[0]+=1
 good=mode=='all' or mode=='mixed' and (rank==0 if world>1 else index<2) or mode=='truncated'
 answer=number if good else 999
 tokens=tokenizer.encode('\\boxed{'+str(answer)+'}')+[2]
 return GeneratedResponse(tokens,[1]*len(tokens),mode!='truncated',len(prompt)+len(tokens)-1,
                          seconds=.001,execution_depths=[1]*len(tokens),cache_policy='exact_lazy_depth')
with patch('ep_loop.training.generate',side_effect=generated):
 train(config,folder,'cpu',smoke=True,resume=resume)
