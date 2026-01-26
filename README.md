# UCX

UCX is a resilient training framework built on [Universal Checkpointing](https://github.com/Supercomputing-System-AI-Lab/UniversalCheckpointing). It makes reconfigurable recovery from checkpoints correct and efficient. It provides two key features: (1) **Irregular world size support** for flexible parallelism configurations, and (2) **Optimized checkpoint conversion** with offset-based reading for reduced I/O overhead.

## Quick Start

### Step 1: Install Dependencies

Install DeepSpeed and dependencies. See [INSTALL_GUIDE_ON_AMD.md](INSTALL_GUIDE_ON_AMD.md) for detailed installation instructions.

### Step 2: Clone this Repo

```bash
git clone https://github.com/Supercomputing-System-AI-Lab/UniversalCheckpointing/
git checkout ucx
cd UniversalCheckpointing
```

### Step 3: Configure Environment

Create `~/.deepspeed_env` file:

```bash
PATH=$HOME/local/bin:$PATH
LD_LIBRARY_PATH=$HOME/local/lib:$LD_LIBRARY_PATH
PDSH_RCMD_TYPE=ssh
PATH="/path/to/conda/env:$PATH"
```

## Usage Examples

### Example 1: Training with Irregular World Size

**Irregular world size** means the total GPU count does not need to satisfy strict divisibility constraints of parallelism configurations. For example, with `GBS=256` and `DP=7`, UCX can assign local batches as `[37, 37, 37, 37, 36, 36, 36]` instead of requiring strict divisibility.

```bash
# Multi-node
python run_single_ds_irregular_devicemesh.py

# Single-node
python run_single_ds_irregular_devicemesh_torchrun_singlenode.py
```

Modify training hyperparameters by editing the configuration inside these scripts.

### Example 2: Optimized UCP Conversion

UCX introduces offset-based reading to accelerate the Universal Checkpoint (UCP) conversion process. Distributed checkpoint (DCP) files are memory-mapped, and only the byte ranges corresponding to requested slices are loaded. This eliminates intermediate files and reduces I/O volume by approximately 50%.

**Prerequisites**: Install `mpi4py`

```bash
sbatch ucx_conversion.slurm
```

**Parameters** in `ucx_conversion.slurm`:
- `IN_FOLDER`: Path to DeepSpeed checkpoint folder
- `OUT_FOLDER`: Output path for universal checkpoint
- `N` (num_workers): Number of parallel workers per node