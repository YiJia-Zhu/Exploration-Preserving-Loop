"""The tested n-gram settings, used only by fixed-depth training rollouts."""


def ngram_speculative_config():
    return dict(method='ngram', num_speculative_tokens=5, prompt_lookup_min=2,
                prompt_lookup_max=5, rejection_sample_method='standard')


def rollout_ngram_enabled(config):
    enabled = config.get('implementation', {}).get('vllm_rollout_ngram', False)
    if type(enabled) is not bool:
        raise ValueError('vllm_rollout_ngram must be boolean')
    if enabled:
        from ep_loop.execution import execution_mode, VLLM_FIXED
        if (config['model']['family'] != 'ouro' or execution_mode(config) != VLLM_FIXED
                or config['implementation'].get('rollout_backend', 'vllm') != 'vllm'):
            raise ValueError('vllm_rollout_ngram requires the Ouro vllm_fixed training route')
    return enabled


def validate_ngram_actor(enabled, policy, teacher_trace, modern=True):
    if type(enabled) is not bool:
        raise ValueError('rollout_ngram must be boolean')
    if enabled and (policy != 'fixed' or teacher_trace):
        raise ValueError('rollout_ngram is only supported by the fixed-depth student rollout actor')
    if enabled and not modern:
        raise ValueError('Training n-gram requires vLLM >=0.26; select the vllm026 interpreter '
                         'or disable VLLM_ROLLOUT_NGRAM')
