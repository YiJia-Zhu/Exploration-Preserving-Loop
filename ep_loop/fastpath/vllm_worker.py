"""Worker extension loaded through vLLM's public extension/RPC interface."""
from pathlib import Path
from datetime import datetime,timezone
import json
import os
import sys
import time
from zoneinfo import ZoneInfo
import torch


class OuroWorkerExtension:
    def ep_begin_teacher_trace(self, output):
        """Capture teacher-forced prefill states on pinned vLLM 0.11.

        The runner always calls this bookkeeping method with all scheduled
        hidden states, including when prompt_logprobs is not requested. This
        instance-local wrapper leaves its original behavior intact.
        """
        model=self.get_model()
        if not model.model.teacher_trace:
            raise ValueError('Teacher RPC requires the all-loop readout engine')
        if getattr(self,'_ep_teacher_original',None) is not None:
            raise RuntimeError('Teacher trace is already active')
        self._ep_teacher_output=Path(output);self._ep_teacher_output.mkdir(parents=True,exist_ok=True)
        self._ep_teacher_chunks={}
        runner=self.model_runner
        original=runner._get_prompt_logprobs_dict
        self._ep_teacher_original=original
        def capture(hidden_states,num_scheduled_tokens):
            # One bulk CPU transfer per scheduled prefill, not per token.
            payload=hidden_states.detach().to('cpu')
            for req_id in runner.input_batch.req_ids:
                count=num_scheduled_tokens[req_id]
                index=runner.input_batch.req_id_to_index[req_id]
                offset=int(runner.query_start_loc.np[index])
                start=int(runner.requests[req_id].num_computed_tokens)
                self._ep_teacher_chunks.setdefault(req_id,[]).append((start,payload[offset:offset+count]))
            return original(hidden_states,num_scheduled_tokens)
        runner._get_prompt_logprobs_dict=capture
        return True

    def ep_end_teacher_trace(self):
        from safetensors.torch import save_file
        original=getattr(self,'_ep_teacher_original',None)
        if original is None:raise RuntimeError('No teacher trace is active')
        self.model_runner._get_prompt_logprobs_dict=original
        self._ep_teacher_original=None
        model=self.get_model();loops=model.model.total_ut_steps;hidden=model.config.hidden_size
        result={}
        try:
            for index,(req_id,chunks) in enumerate(self._ep_teacher_chunks.items()):
                length=max(start+len(chunk) for start,chunk in chunks)
                packed=chunks[0][1].new_empty((length,loops*hidden+loops))
                covered=torch.zeros(length,dtype=torch.bool)
                executed=0
                for start,chunk in chunks:
                    packed[start:start+len(chunk)]=chunk;covered[start:start+len(chunk)]=True
                    executed+=len(chunk)*loops
                if not covered.all():raise RuntimeError('Teacher trace has missing prefill positions')
                states=packed[:,:loops*hidden].reshape(length,loops,hidden).permute(1,0,2).contiguous()
                gates=packed[:,loops*hidden:].T.unsqueeze(-1).contiguous()
                path=self._ep_teacher_output/f'request_{index:06d}.safetensors'
                save_file(dict(states=states,gates=gates),str(path))
                result[req_id]=dict(path=str(path),input_positions=length,executed_token_loops=executed)
        finally:self._ep_teacher_chunks={}
        return result

    def ep_start_trace(self):
        model=self.get_model()
        self._ep_depth_rows=[]
        self._ep_trace_counts={}
        self._ep_progress_clock=time.monotonic()
        def capture(depths):
            request_ids=list(self.model_runner.input_batch.req_ids)
            if len(request_ids)!=depths.shape[0]:
                raise RuntimeError('Depth diagnostics require one logits row per request; no speculative decoding or prompt logprobs')
            # vLLM also computes logits for partial prefills, then discards
            # their sampled tokens. Apply its existing CPU discard mask only
            # to diagnostics, without touching logits, sampling or scheduling.
            discarded=set(self.model_runner.discard_request_indices.np[
                :self.model_runner.num_discarded_requests].tolist())
            self._ep_depth_rows.append((request_ids,depths.detach().clone(),discarded))
            for index,request_id in enumerate(request_ids):
                if index in discarded:continue
                self._ep_trace_counts[request_id]=self._ep_trace_counts.get(request_id,0)+1
            now=time.monotonic()
            if now-self._ep_progress_clock>=10:
                print(json.dumps(dict(event='vllm_generation_progress',
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    local_time=datetime.now(timezone.utc).astimezone(ZoneInfo(os.environ.get('EP_LOOP_LOG_TIMEZONE','Asia/Shanghai'))).isoformat(),
                    generated_tokens=self._ep_trace_counts)),file=sys.stderr,flush=True)
                self._ep_progress_clock=now
        model._depth_callback=capture
        return True

    def ep_finish_trace(self):
        self.get_model()._depth_callback=None
        rows=getattr(self,'_ep_depth_rows',[])
        result={}
        if rows:
            flat=torch.cat([values for _,values,_ in rows]).float().cpu().tolist()
            offset=0
            for request_ids,values,discarded in rows:
                for index,request_id in enumerate(request_ids):
                    if index in discarded:continue
                    result.setdefault(request_id,dict(depths=[],gate_logits=[]))
                    result[request_id]['depths'].append(int(flat[offset+index][0]))
                    result[request_id]['gate_logits'].append(flat[offset+index][1:])
                offset+=len(request_ids)
        self._ep_depth_rows=[]
        return result

    def ep_load_weights(self, filename, version):
        """Called while idle, after wake-up and before any next-update request."""
        from safetensors import safe_open
        if int(version)<=getattr(self,'_ep_weight_version',-1):
            raise ValueError('Student weight versions must increase monotonically')
        with safe_open(str(Path(filename)),framework='pt',device='cpu') as source:
            names=list(source.keys())
            loaded=self.get_model().load_weights((name,source.get_tensor(name)) for name in names)
        self._ep_weight_version=int(version)
        return dict(weight_version=int(version),source_tensors=len(names),loaded_parameters=len(loaded))

    def ep_health(self):
        return dict(weight_version=getattr(self,'_ep_weight_version',0),
                    model_type=type(self.get_model()).__name__,
                    configured_recurrent_steps=int(self.get_model().model.total_ut_steps))
