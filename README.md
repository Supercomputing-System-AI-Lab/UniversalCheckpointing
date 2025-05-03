## 📢 News

Universal Checkpointing has been adopted by prominent organizations and research institutions for model pre-training and fine-tuning. Here are some notable adopters:

* [BigScience - BLOOM 176B](https://huggingface.co/bigscience/bloom)
* [Microsoft Phi-3.5-MoE 42B](https://huggingface.co/microsoft/Phi-3.5-MoE-instruct)
* [University of California, Berkeley - SmileyLlama 8B](https://arxiv.org/abs/2409.02231)
* [Renmin University of China - YuLan-Mini 2.4B](https://arxiv.org/abs/2412.17743)

Moreover, Universal Checkpointing has been tested on:

* AMD cluster. AMD will share experience at [FMS’25](https://futurememorystorage.com): “Flexible, Efficient, Resilient Training on AMD GPUs with DeepSpeed UCP”
* [Aurora Intel GPU cluster](https://www.alcf.anl.gov/aurora) by [Argonne National Lab](https://www.anl.gov)

## 📙 About

Universal Checkpointing (UCP) is a novel checkpointing system that enables flexible and efficient model training with reconfigurable parallelism.

Why Universal Checkpointing?

- ✨ **Comprehensive Parallelism Support**: Supports DP, ZeRO-DP (Stage 1/2/3), PP, TP, SP, and any combination of these parallel strategies.
- ✨ **Versatile Model Architecture**: Compatible with Dense, MoE, and GQA model architectures.
- ✨ **Optimized Performance**: Efficient checkpointing with minimal overhead and maximum flexibility.
- ✨ **Seamless Integration**: No code refactoring required - works with existing training pipelines.

Want to know more details? Read our paper & blog!

- **Paper**: [ATC'25 paper](https://www.usenix.org/conference/atc25)
- **Blog**: [DeepSpeed](https://www.deepspeed.ai/tutorials/universal-checkpointing/), [Megatron-DeepSpeed](https://github.com/deepspeedai/Megatron-DeepSpeed/blob/main/examples_deepspeed/universal_checkpointing/README.md), [SSAIL](https://supercomputing-system-ai-lab.github.io/projects/ucp)
