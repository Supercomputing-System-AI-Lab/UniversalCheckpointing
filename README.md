## 📢 News

Universal Checkpointing has been adopted by prominent organizations and research institutions for model pre-training and fine-tuning. Here are some notable adopters:

* [BigScience - BLOOM 176B](https://huggingface.co/bigscience/bloom)
* [Microsoft - Phi-3.5-MoE 42B](https://huggingface.co/microsoft/Phi-3.5-MoE-instruct)
* [University of California, Berkeley - SmileyLlama 8B](https://arxiv.org/abs/2409.02231)
* [Renmin University of China - YuLan-Mini 2.4B](https://arxiv.org/abs/2412.17743)

Moreover, Universal Checkpointing has been tested on:

* **AMD** cluster by AMD researchers. AMD will share experience at [FMS’25](https://futurememorystorage.com): “Flexible, Efficient, Resilient Training on AMD GPUs with DeepSpeed UCP”
* [Aurora **Intel** GPU cluster](https://www.alcf.anl.gov/aurora) by [Argonne National Lab](https://www.anl.gov)

## 📙 About

Universal Checkpointing (UCP) is a novel checkpointing system that enables flexible and efficient model training with reconfigurable parallelism.

Why Universal Checkpointing?

- ✨ **Comprehensive Parallelism Support**: Supports DP, ZeRO-DP (Stage 1/2/3), PP, TP, SP, and any combination of these parallel strategies.
- ✨ **Versatile Model Architecture**: Compatible with Dense, MoE, and GQA model architectures.
- ✨ **Optimized Performance**: Efficient reconfiguration with minimal overhead.
- ✨ **Seamless Integration**: No code refactoring required, orthogonal to existing checkpoint saving techniques.

Want to know more details? Read our paper & blogs!

- **Paper**: [ATC'25 paper](https://www.usenix.org/conference/atc25)
- **Blog**: [DeepSpeed](https://www.deepspeed.ai/tutorials/universal-checkpointing/), [Megatron-DeepSpeed](https://github.com/deepspeedai/Megatron-DeepSpeed/blob/main/examples_deepspeed/universal_checkpointing/README.md), [SSAIL](https://supercomputing-system-ai-lab.github.io/projects/ucp)


## 🔥 Quick Start

### Install the required dependencies

Install DeepSpeed
```
git clone https://github.com/xylian86/DeepSpeed.git
cd DeepSpeed
git checkout ucp
pip install -e .
```

Install Apex
```
git clone https://github.com/NVIDIA/apex.git
cd apex
pip install --global-option="--cpp_ext" --global-option="--cuda_ext" --no-cache -v --disable-pip-version-check --no-build-isolation .  2>&1 | tee build.log
```

### Tiny example (2 GPUs)




### Correctness Check when converting the 


### Efficiency Check 


## 📜 Citation

```bibtex
@inproceedings{ucp,
  title = {Universal Checkpointing: A Flexible and Efficient Distributed Checkpointing System for Large-Scale DNN Training with Reconfigurable Parallelism},
  author = {Lian, Xinyu and Jacobs, Sam Ade and Kurilenko, Lev and Tanaka, Masahiro and Bekman, Stas and Ruwase, Olatunji and Zhang, Minjia},
  booktitle = {2025 USENIX Annual Technical Conference},
  year = {2025}
}
```

## 🙏 Acknowledgement

- [DeepSpeed](https://github.com/deepspeedai/DeepSpeed)
