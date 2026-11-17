# LoopOPD训练规模对齐

当前三个启动脚本默认32题×8回答＝256条／更新，目标100次更新。每组目标25,600条fresh on-policy回答、3,200次题目抽取。全局batch后仅更新一次；两卡各128条，vLLM按rank一次提交全部请求，未设置max_num_seqs。

TRAIN_EXAMPLES=all：17391题公开源中17道prompt超长，剩17374题按seed42打乱；17310训练、64开发评测，内容无交集。旧256题划分仍保留。当前采样批内无放回、跨批可重复；100步题目采样推演覆盖2923道不同题，尚未执行这些训练更新。

保留16384长度、DAPO64题×1评测、seed42与统一入口10小时上限。100步是目标，10小时可能先到。论文31744长度、AIME16次评测、4张H200与本地设置不同。跨批采样器未核实与作者一致，不宣称完整论文复现。

教师临时状态默认放在/dev/shm内存盘，按回答读入GPU后删除，使用前检查空间。两卡满长度256条的状态上限约68GiB。checkpoint仍保存在OUTPUT_ROOT，默认滚动保留一份完整恢复点及最终快照。

验证：114项CPU测试通过；三路线默认计划通过；新完整数据池与旧数字题池准备通过；模拟引擎验证两卡各128条一次提交。此次未启动GPU训练，没有batch256长生成耗时或有效性结果。首次回归失败记录保留于cpu_tests_initial.log，修正后结果见cpu_tests.log。

参数与检查摘要见validation.json；数据与抽题推演见data_and_protocol_validation.json；准备好的top_n配置见prepared/full_pool_32x8_100_top_n/configs/ours.json。
