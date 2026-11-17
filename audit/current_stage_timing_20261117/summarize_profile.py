"""Summarize one real update, separating engine cold starts from batch work."""
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
launch=json.loads((Path(__file__).with_name('latest_profile.json')).read_text())
output=Path(launch['output']);panel=Path(launch['panel'])
summary_path=output/'summary.json'
if not summary_path.exists():raise SystemExit('The full-length update is still running')
summary=json.loads(summary_path.read_text())
assert summary['completed_updates']==1
metrics=[json.loads(l) for l in (output/'metrics.jsonl').read_text().splitlines()]
assert len(metrics)==1 and metrics[0]['responses']==16
ranks={};finishes=[]
for rank in (0,1):
    rows=[json.loads(l) for l in (output/f'events_rank{rank}.jsonl').read_text().splitlines()]
    seconds=defaultdict(float)
    for e in rows:
        if e['event']=='stage_end' and e.get('update')==1:seconds[e['stage']]+=e['seconds']
    initialized=[json.loads(l) for l in (output/f'engine_initialization_rank{rank}.jsonl').read_text().splitlines()]
    initialization={e['kind']:e['seconds'] for e in initialized}
    stages=dict(seconds)
    stages['student_generation_excluding_engine_init']=stages['vllm_on_policy_batch']-initialization['student']
    stages['teacher_labeling_excluding_engine_init']=stages['vllm_teacher_label_batch']-initialization['teacher']
    ranks[str(rank)]=dict(stage_seconds=stages,engine_initialization_seconds=initialization)
    finishes.extend(e['timestamp'] for e in rows if e['event']=='run_end')
assert len(finishes)==2
end=max(datetime.fromisoformat(t) for t in finishes)
start=datetime.fromisoformat(launch['started_at'])
checkpoint=output/'checkpoints/update_000001'
assert (checkpoint/'checkpoint.complete.json').exists()
report=dict(scope='One real update; global16, actual EOS/16384 generation, frozenK4 teacher, fixedN2 student; not an efficacy result',
    global_response_tokens=metrics[0]['response_tokens'],mean_response_tokens=metrics[0]['response_tokens']/16,
    truncated_responses=metrics[0]['truncated_responses'],loss=metrics[0]['mean_response_loss'],
    teacher_exit_hist=metrics[0]['exit_hist'],teacher_backend=metrics[0]['teacher_backend'],
    parent_wall_seconds=(end-start).total_seconds(),rank_details=ranks,
    checkpoint=str(checkpoint),checkpoint_bytes=sum(f.stat().st_size for f in checkpoint.rglob('*') if f.is_file()),
    caveat='Rank-local host timings include waits and nested stages. The two ranks overlap, so do not sum their times. Subtracting initialization estimates warm engine work, not a second full-length update measurement.')
(panel/'stage_timing.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
labels=[('CPU卸载HF学生／optimizer','training_state_offload'),('vLLM学生引擎首次初始化','student_init'),
        ('vLLM rollout（扣除引擎初始化，含权重同步／sleep）','student_generation_excluding_engine_init'),
        ('vLLM教师引擎首次初始化','teacher_init'),('vLLM教师标注（扣除引擎初始化）','teacher_labeling_excluding_engine_init'),
        ('恢复HF训练状态／DDP','training_state_restore'),('教师状态读取','teacher_forward'),
        ('JS阶段选择','stage_planning'),('HF学生前向','student_replay'),('KL损失前向','loss_readout'),
        ('反向传播（含DDP等待）','backward'),('AdamW更新','optimizer_step'),('完整checkpoint保存','checkpoint_save'),
        ('单次更新合计（含两引擎首次初始化）','update')]
lines=['# 当前vLLM流程：一次完整长度更新计时','',
       f"实测输出：{output}。全局16条回答，生成{report['global_response_tokens']:,} tokens，平均{report['mean_response_tokens']:.0f} tokens／回答，{report['truncated_responses']}条到长度上限。",'',
       '| 阶段 | rank0秒 | rank1秒 |','|---|---:|---:|']
for label,key in labels:
    values=[]
    for rank in ['0','1']:
        r=ranks[rank]
        value=r['engine_initialization_seconds'][key.removesuffix('_init')] if key in ('student_init','teacher_init') else r['stage_seconds'][key]
        values.append(f'{value:.2f}')
    lines.append('| '+label+' | '+' | '.join(values)+' |')
lines.extend(['',f"从启动到训练进程退出约{report['parent_wall_seconds']/60:.2f}分钟。完整checkpoint占{report['checkpoint_bytes']/2**30:.2f}GiB，包含optimizer和双rank RNG，可恢复。",'',
    '两卡并行，不能相加rank0和rank1；各细分阶段为主机墙钟，反向含同步等待。扣除首次引擎初始化的数值用于估计热引擎工作量，不等于另一次完整长度更新实测。此次只跑一个更新，不含最终64题评测。'])
(panel/'stage_timing.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
