import os
import subprocess


def generate_log_path(config, base_dir: str = "logs/sigcomm26") -> tuple:
    """
    Generate log directory and filename based on config.
    
    Format:
    - Directory: {base_dir}/{model_size}_gbs{global_batch_size}_1nodes
    - Filename: pretrain_gpt_gbs_{gbs}_gas{gas}_sz_{size}_tp{tp}_pp{pp}_dp{dp}_ZeRO{zero}_gpn{gpus_per_node}.out
    
    Args:
        config: Configuration dictionary
        base_dir: Base directory for logs
        
    Returns:
        Tuple of (log_dir, log_filename)
    """
    model_size = config['model_size']
    gbs = config['global_batch_size']
    gas = config['grad_accum_steps']
    tp = config['tensor_parallel']
    pp = config['pipeline_parallel']
    dp = config['data_parallel']
    zero = config['zero_stage']
    gpn = config['gpus_per_node']
    ac = config.get('activation_checkpointing', 1)
    
    # Directory: logs/sigcomm26/3B_gbs64_1nodes
    log_dir = f"{base_dir}/{model_size}_gbs{gbs}_1nodes"
    
    # Filename: pretrain_gpt_gbs_64_gas8_sz_3B_tp1_pp1_dp4_ZeRO3_gpn4.out
    # Add _noac suffix if activation_checkpointing=0
    ac_suffix = "_noac" if ac == 0 else ""
    log_filename = f"pretrain_gpt_gbs_{gbs}_gas{gas}_sz_{model_size}_tp{tp}_pp{pp}_dp{dp}_ZeRO{zero}_gpn{gpn}{ac_suffix}.out"
    
    return log_dir, log_filename


def build_env_string(config):
    """Build environment variable export string for SSH command."""
    env_vars = {
        'TP': str(config['tensor_parallel']),
        'PP': str(config['pipeline_parallel']),
        'DP': str(config['data_parallel']),
        'SP': str(config['sequence_parallel']),
        'GLOBAL_BATCH': str(config['global_batch_size']),
        'GRAD_ACCUM_STEPS': str(config['grad_accum_steps']),
        'ZERO_STAGE': str(config['zero_stage']),
        'SIZE_TAG': str(config['model_size']),
        'GPUS_PER_NODE': str(config['gpus_per_node']),
        'ACTIVATION_CHECKPOINTING': str(config['activation_checkpointing']),
        'EXIT_INTERVAL': str(config['exit_interval']),
        'MASTER_PORT': str(config.get('master_port', 13701)),
    }
    
    # Build export statements
    exports = ' '.join([f"export {k}={v};" for k, v in env_vars.items()])
    return exports


def get_script_for_cluster(cluster: str) -> str:
    """Get the appropriate shell script for the given cluster."""
    scripts = {
        'delta': 'pretrain_gpt_on_delta_torchrun_salloc.sh',
        'amd': 'pretrain_gpt_on_amd_torchrun_salloc.sh',
    }
    if cluster not in scripts:
        raise ValueError(f"Unknown cluster '{cluster}'. Supported: {list(scripts.keys())}")
    return scripts[cluster]


def run_single_node_torchrun(config):
    """Run single-node training via SSH + torchrun."""
    node = config.get('node')
    if not node:
        raise ValueError("'node' must be specified (e.g., 'gpua001')")
    
    cluster = config.get('cluster', 'delta')
    script_name = get_script_for_cluster(cluster)
    
    print(f"Target node: {node}")
    print(f"Cluster: {cluster}")
    
    # Get current working directory (absolute path)
    workdir = os.getcwd()
    
    # Set log directory and filename
    log_base_dir = config.get('log_base_dir', 'logs/sigcomm26')
    log_dir, log_filename = generate_log_path(config, base_dir=log_base_dir)
    log_dir_abs = os.path.abspath(log_dir)
    log_path = os.path.join(log_dir_abs, log_filename)
    
    # Build environment variables string
    env_exports = build_env_string(config)
    
    # Add log-related environment variables
    env_exports += f" export LOG_DIR={log_dir_abs};"
    env_exports += f" export LOG_FILENAME={log_filename};"
    env_exports += " export LOG_TO_FILE=1;"
    
    print("=" * 60)
    print("Single-Node Torchrun Configuration:")
    print(f"  Cluster: {cluster}")
    print(f"  Node: {node}")
    print(f"  GPUS_PER_NODE: {config['gpus_per_node']}")
    print(f"  Model: {config['model_size']}")
    print(f"  Parallelism: TP={config['tensor_parallel']}, PP={config['pipeline_parallel']}, "
          f"DP={config['data_parallel']}, SP={config['sequence_parallel']}")
    print(f"  ZeRO Stage: {config['zero_stage']}")
    print(f"  Global Batch: {config['global_batch_size']}")
    print(f"  LOG_PATH: {log_path}")
    print(f"  Script: {script_name}")
    print("=" * 60)
    
    # Build SSH command
    remote_cmd = f"cd {workdir} && {env_exports} bash {script_name}"
    cmd = ['ssh', node, remote_cmd]
    
    print(f"Running on {node}...")
    print(f"Remote command: {remote_cmd[:100]}...")
    
    # Use Popen to stream output in real-time
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1  # Line buffered
    )
    
    # Stream output
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)
    
    print(f"\nLog saved to: {log_path}")


if __name__ == "__main__":
    # Config for single-node torchrun
    config = {
        'cluster': 'delta',  # 'delta' or 'amd'
        'node': 'gpua012',  # Single node name
        'tensor_parallel': 1,
        'pipeline_parallel': 1,
        'data_parallel': 4,
        'sequence_parallel': 1,
        'global_batch_size': 64,
        'grad_accum_steps': 4,
        'zero_stage': 3,
        'model_size': '1B',
        'gpus_per_node': 4,
        'activation_checkpointing': 0,
        'master_port': 13701,
        'log_base_dir': 'logs/xxx',
        'exit_interval': 10,
    }
    
    run_single_node_torchrun(config)
