"""Exactly 20 explicit teacher probabilities plus a uniform residual tail.

Only the teacher distribution is approximated: preserve its top-k original
probabilities and spread the remaining mass uniformly over omitted tokens.
JS is exact for these reconstructed distributions. Distillation remains full
student-to-teacher reverse KL, using an analytic tail cross-entropy term.
"""
from dataclasses import dataclass
import math
import torch


@dataclass
class TopKDistribution:
    token_ids: torch.Tensor
    log_probs: torch.Tensor
    log_tail_mass: torch.Tensor
    vocabulary: int

    @classmethod
    def from_logits(cls, logits, top_k=20):
        if top_k < 1:
            raise ValueError('top_k must be positive')
        logits=logits.float()
        vocabulary=logits.shape[-1]
        values,indices=logits.topk(min(top_k,vocabulary),dim=-1,sorted=True)
        log_probs=values-logits.logsumexp(-1,keepdim=True)
        mass=log_probs.exp().sum(-1)
        tail=(1-mass).clamp(min=torch.finfo(log_probs.dtype).tiny,max=1)
        if indices.shape[-1]==vocabulary:tail=torch.zeros_like(tail)
        return cls(indices.to(torch.int32),log_probs,tail.log(),vocabulary)

    def slice(self, start, end):
        return type(self)(self.token_ids[:,start:end],self.log_probs[:,start:end],
                          self.log_tail_mass[:,start:end],self.vocabulary)

    def uniform_tail_log_prob(self):
        omitted=self.vocabulary-self.token_ids.shape[-1]
        return self.log_tail_mass-math.log(omitted) if omitted else torch.zeros_like(self.log_tail_mass)

    def dense_log_probs(self):
        """Reference/debug only; production losses never persist this tensor."""
        result=self.uniform_tail_log_prob()[...,None].expand(*self.log_probs.shape[:-1],self.vocabulary).clone()
        return result.scatter(-1,self.token_ids.long(),self.log_probs)


def _probability_at(query_ids, ids, log_probs, tail_log_prob):
    equal=query_ids[...,None]==ids[...,None,:]
    where=equal.to(torch.int64).argmax(-1)
    return torch.where(equal.any(-1),log_probs.gather(-1,where),tail_log_prob[...,None]).exp()


def adjacent_js(sparse):
    """Union-of-top-k + one shared outside bucket; no missing-token zero fill."""
    ids=sparse.token_ids
    if ids.shape[0]<2:return sparse.log_probs.new_empty((0,ids.shape[1]))
    union=torch.cat((ids[:-1],ids[1:]),-1)
    same=union[...,None]==union[...,None,:]
    previous=torch.ones(same.shape[-2:],device=ids.device,dtype=torch.bool).tril(-1)
    unique=~(same & previous).any(-1)
    tail=sparse.uniform_tail_log_prob()
    left=_probability_at(union,ids[:-1],sparse.log_probs[:-1],tail[:-1])*unique
    right=_probability_at(union,ids[1:],sparse.log_probs[1:],tail[1:])*unique
    outside=sparse.vocabulary-unique.sum(-1)
    omitted=sparse.vocabulary-ids.shape[-1]
    left_other=outside*sparse.log_tail_mass[:-1].exp()/max(1,omitted)
    right_other=outside*sparse.log_tail_mass[1:].exp()/max(1,omitted)
    left=torch.cat((left,left_other[...,None]),-1)
    right=torch.cat((right,right_other[...,None]),-1)
    midpoint=(left+right)*.5
    tiny=torch.finfo(left.dtype).tiny
    kl=lambda value:torch.special.xlogy(value,value/midpoint.clamp_min(tiny)).sum(-1)
    return ((kl(left)+kl(right))*.5).clamp(min=0,max=math.log(2))


def stage_reverse_kl(student_logits, teacher, plan):
    """Original full-vocabulary KL(p_student || reconstructed q_teacher).

    The student's tail entropy/probability remains in the objective; this is
    neither forward KL nor cross-entropy nor a top-k-renormalized student loss.
    """
    logp=student_logits.float().log_softmax(-1)
    p=logp.exp()
    entropy_term=(p*logp).sum(-1)
    positions=torch.arange(logp.shape[1],device=logp.device)
    losses=logp.new_zeros(logp.shape[1])
    for slot in range(logp.shape[0]):
        selected=(plan.teacher_depths[slot]-1).clamp_min(0)
        ids=teacher.token_ids[selected,positions].long()
        q=teacher.log_probs[selected,positions]
        mass=p[slot].gather(-1,ids)
        tail_mass=(1-mass.sum(-1)).clamp_min(0)
        tail_log=teacher.uniform_tail_log_prob()[selected,positions]
        cross=(mass*q).sum(-1)+tail_mass*tail_log
        kl=entropy_term[slot]-cross
        losses+=plan.weights[slot].to(kl)*kl
    return losses
