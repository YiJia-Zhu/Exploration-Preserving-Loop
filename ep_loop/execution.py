"""Explicit execution choices; the default retains the full dynamic method."""

HF_DYNAMIC = 'hf_dynamic'
VLLM_TOP20 = 'vllm_top20'
HF_TOP20 = 'hf_top20'
VLLM_FIXED = 'vllm_fixed'
HF_ADAPTIVE = 'hf_adaptive'
HYBRID = 'hybrid'
MODES = (HF_DYNAMIC, HF_ADAPTIVE, HYBRID, VLLM_TOP20, HF_TOP20, VLLM_FIXED)
VLLM_MODES = (VLLM_TOP20, VLLM_FIXED)
HF_TRAIN_MODES = (HF_DYNAMIC, HF_ADAPTIVE, HYBRID)
VLLM_EVAL_MODES = (*VLLM_MODES, HYBRID)


def execution_mode(config):
    implementation = config.get('implementation', {})
    algorithm = config.get('algorithm', {})
    mode = implementation.get('execution_mode')
    # Preserve previously saved opt-in configs without relabeling their method.
    if mode is None:
        if algorithm.get('teacher_distribution') == 'top20_original_mass_uniform_tail':
            mode = VLLM_TOP20 if implementation.get('rollout_backend', 'vllm') == 'vllm' else HF_TOP20
        else:
            mode = HF_DYNAMIC
    if mode not in MODES:
        raise ValueError(f'Unknown execution_mode: {mode}; choose one of {MODES}')
    return mode


def validate_execution(config):
    mode = execution_mode(config)
    implementation = config.get('implementation', {})
    algorithm = config.get('algorithm', {})
    backend = 'vllm' if mode in VLLM_MODES else 'hf'
    if implementation.get('rollout_backend', backend) != backend:
        raise ValueError(f'{mode} requires rollout_backend={backend}')
    if mode in {*HF_TRAIN_MODES, VLLM_FIXED}:
        if algorithm.get('teacher_top_k') is not None or algorithm.get('teacher_distribution', 'full_vocab') != 'full_vocab':
            raise ValueError(f'{mode} requires full-vocabulary JS/KL; choose hf_top20 or vllm_top20 explicitly for approximation')
    else:
        if algorithm.get('teacher_top_k') != 20 or algorithm.get('teacher_distribution') != 'top20_original_mass_uniform_tail':
            raise ValueError(f'{mode} requires explicit teacher_top_k=20 and top20 distribution metadata')
    policy=algorithm.get('student_policy','fixed')
    if mode==VLLM_FIXED:
        if policy not in {'fixed','native'} or algorithm.get('teacher_endpoint') not in {'fixed','native'}:
            raise ValueError('vllm_fixed requires fixed student depth and full-K teacher computation')
        if policy=='fixed' and algorithm.get('stage_selection') not in {'endpoint','top_n','uniform_n'}:
            raise ValueError('vllm_fixed uses endpoint, top_n or uniform_n supervision')
    if mode==HF_ADAPTIVE and policy=='learned' and algorithm.get('teacher_endpoint')!='native_adaptive':
        raise ValueError('hf_adaptive requires real native-CDF teacher execution')
    if mode==HYBRID and policy=='learned':
        evaluation=config.get('evaluation',{})
        if evaluation.get('student_policy')!='fixed':
            raise ValueError('hybrid requires fixed-depth deployment evaluation')
        budgets=evaluation.get('student_loops',[])
        if not budgets or any(type(depth) is not int or not 1<=depth<=algorithm['teacher_max_loops'] for depth in budgets):
            raise ValueError('Hybrid deployment budgets must lie in [1,K]')
    return mode
