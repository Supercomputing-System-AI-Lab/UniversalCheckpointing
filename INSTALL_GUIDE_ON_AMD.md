# Installation for Megatron-DeepSpeed on AMD GPUs

This instruction guides you through the installation of the **Megatron-DeepSpeed** on the AMD GPUs, and finally trains a toy llm with it.

## Dependencies

- **Python**: 3.9
- **PyTorch**: 2.6.0+rocm6.2.4
- **DeepSpeed**: [`241bffde`](https://github.com/deepspeedai/DeepSpeed/commit/241bffde)
- **APEX**: [ROCm version](https://github.com/ROCm/apex) for mixed precision training

## 1. Create and Activate Conda Environment

```bash
conda create -n ucx python=3.9 -y
conda activate ucx
```

## 2. Install Required Libraries

- [PyTorch](https://pytorch.org/)
- [APEX (ROCm)](https://github.com/ROCm/apex)

Follow [here](https://github.com/ROCm/apex?tab=readme-ov-file#supported-versions) to clone and checkout proper APEX branch given your torch version.

```bash
git clone https://github.com/ROCm/apex.git
git checkout release/1.6.0 #Follow https://github.com/ROCm/apex?tab=readme-ov-file#supported-versions to choose proper branch matching your torch version.
module load rocm/6.2.1  # align with the rocm that pytorch was built on
cd apex
export MAX_JOBS=6  # avoid your system being overloaded
pip install -v --no-build-isolation --config-settings "--build-option=--cpp_ext" ./  #  I omitted --config-settings "--build-option=--cuda_ext" as it will raise an error while compiling `_apex_nccl_allocator`. Also mentioned at issue#191: https://github.com/ROCm/apex/issues/191
```

- [Deepspeed](https://www.deepspeed.ai/tutorials/advanced-install/)
```bash
git clone https://github.com/deepspeedai/DeepSpeed.git deepspeed
cd deepspeed
git checkout -b ucp-amd-base 241bffde
pip install .
```
- Others: transformers six pybind11
```bash
pip install transformers six pybind11
```

## 3. Install PDSH Without SSH

```bash
git clone https://github.com/chaos/pdsh.git
cd pdsh
./bootstrap
mkdir -p $HOME/local
./configure --prefix=$HOME/local --with-ssh
make -j$(nproc)
make install
```

## 4. Set Up .deepspeed_env

Create a file ~/.deepspeed_env with the content:

```bash
PATH=$HOME/local/bin:$PATH
LD_LIBRARY_PATH=$HOME/local/lib:$LD_LIBRARY_PATH
PDSH_RCMD_TYPE=ssh
PATH="/path/to/your/conda/env/bin:$PATH"  # replace with your conda env path.
```

## 6. Test

Test with training a toy llm with megatron-deepspeed.

```bash
sbatch run_bf16_debug.sh
```

Check the log file inside `slurm_logs/`, if works successfully, it will print loss and save checkpoints into `exp_dir/`.