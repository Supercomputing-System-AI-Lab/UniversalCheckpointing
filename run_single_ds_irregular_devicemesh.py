import os
import re
import subprocess
from datetime import datetime


# Cluster-specific configurations
CLUSTER_CONFIG = {
    'delta': {
        'slurm_script': 'pretrain_gpt_on_delta_ds_irregular_devicemesh.slurm',
        'salloc_script': 'pretrain_gpt_ds_irregular_devicemesh_salloc.sh',
        'training_script': 'pretrain_gpt_on_delta_ds_irregular_devicemesh.sh',
    },
    'amd': {
        'slurm_script': 'pretrain_gpt_on_amd_ds_irregular_devicemesh.slurm',
        'salloc_script': 'pretrain_gpt_ds_irregular_devicemesh_salloc.sh',
        'training_script': 'pretrain_gpt_on_amd_ds_irregular_devicemesh.sh',
    },
}


def parse_slurm_nodes(node_spec: str) -> list:
    """
    Parse slurm node name format into a list of node names.
    
    Supports formats:
    - Slurm bracket notation: gpua[020,046,051,089] -> ['gpua020', 'gpua046', 'gpua051', 'gpua089']
    - Slurm range notation: gpua[001-004] -> ['gpua001', 'gpua002', 'gpua003', 'gpua004']
    - Mixed notation: gpua[001-003,005] -> ['gpua001', 'gpua002', 'gpua003', 'gpua005']
    - Comma-separated: gpua020,gpua046 -> ['gpua020', 'gpua046']
    - Single node: gpua020 -> ['gpua020']
    - AMD mixed format: k002-006,k003-[001-003] -> ['k002-006', 'k003-001', 'k003-002', 'k003-003']
    
    Args:
        node_spec: Node specification string in slurm format
        
    Returns:
        List of expanded node names
    """
    node_spec = node_spec.strip()
    
    # Split by comma, but not commas inside brackets
    # Use regex to find all parts (either bracket notation or plain names)
    parts = []
    current = ""
    bracket_depth = 0
    
    for char in node_spec:
        if char == '[':
            bracket_depth += 1
            current += char
        elif char == ']':
            bracket_depth -= 1
            current += char
        elif char == ',' and bracket_depth == 0:
            if current.strip():
                parts.append(current.strip())
            current = ""
        else:
            current += char
    
    if current.strip():
        parts.append(current.strip())
    
    # Parse each part
    nodes = []
    for part in parts:
        nodes.extend(_parse_single_node_spec(part))
    
    return nodes


def _parse_single_node_spec(node_spec: str) -> list:
    """
    Parse a single node specification (no top-level commas).
    
    Handles:
    - Bracket notation: k003-[001-003] -> ['k003-001', 'k003-002', 'k003-003']
    - Plain node: k002-006 -> ['k002-006']
    """
    node_spec = node_spec.strip()
    
    # Check for bracket notation: prefix[items]
    bracket_match = re.match(r'^([a-zA-Z0-9_-]+)\[([^\]]+)\]$', node_spec)
    
    if bracket_match:
        prefix = bracket_match.group(1)
        items = bracket_match.group(2)
        
        nodes = []
        for item in items.split(','):
            item = item.strip()
            # Check if item is a range (e.g., 001-004)
            range_match = re.match(r'^(\d+)-(\d+)$', item)
            if range_match:
                start = int(range_match.group(1))
                end = int(range_match.group(2))
                # Preserve leading zeros
                width = len(range_match.group(1))
                for i in range(start, end + 1):
                    nodes.append(f"{prefix}{str(i).zfill(width)}")
            else:
                # Single item
                nodes.append(f"{prefix}{item}")
        return nodes
    
    # Single node (no brackets)
    return [node_spec]


def generate_hostfile(nodes: list, gpus_per_node_irregular: str, output_dir: str = "Megatron-DeepSpeed/hostfiles") -> str:
    """
    Generate a hostfile for DeepSpeed distributed training.
    
    Args:
        nodes: List of node names
        gpus_per_node_irregular: Comma-separated string of GPU counts per node (e.g., "4,4,4,4")
        output_dir: Directory to save the hostfile
        
    Returns:
        Path to the generated hostfile
    """
    # Parse GPU counts
    gpu_counts = [int(x.strip()) for x in gpus_per_node_irregular.split(',')]
    
    if len(gpu_counts) != len(nodes):
        raise ValueError(
            f"Number of GPU counts ({len(gpu_counts)}) must match number of nodes ({len(nodes)}). "
            f"Nodes: {nodes}, GPU counts: {gpu_counts}"
        )
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate unique filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hostfile_path = os.path.join(output_dir, f"hostfile_salloc_{timestamp}")
    
    # Write hostfile
    with open(hostfile_path, 'w') as f:
        for node, gpu_count in zip(nodes, gpu_counts):
            f.write(f"{node} slots={gpu_count}\n")
    
    print(f"Hostfile created: {hostfile_path}")
    return hostfile_path


def generate_log_path(config, num_nodes: int, base_dir: str = "logs/xxx") -> tuple:
    """
    Generate log directory and filename based on config.
    
    Format:
    - Directory: {base_dir}/{model_size}_gbs{global_batch_size}_{num_nodes}nodes
    - Filename: pretrain_gpt_gbs_{gbs}_gas{gas}_sz_{size}_tp{tp}_pp{pp}_dp{dp}_ZeRO{zero}_dm{device_mesh}.out
    
    Args:
        config: Configuration dictionary
        num_nodes: Number of nodes
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
    dm = config['gpus_per_node_irregular']
    ac = config.get('activation_checkpointing', 1)
    
    # Directory: logs/xxx/3B_gbs64_4nodes
    log_dir = f"{base_dir}/{model_size}_gbs{gbs}_{num_nodes}nodes"
    
    # Filename: pretrain_gpt_gbs_64_gas8_sz_3B_tp1_pp1_dp8_ZeRO0_dm2,2,2,2.out
    # Add _noac suffix if activation_checkpointing=0
    ac_suffix = "_noac" if ac == 0 else ""
    log_filename = f"pretrain_gpt_gbs_{gbs}_gas{gas}_sz_{model_size}_tp{tp}_pp{pp}_dp{dp}_ZeRO{zero}_dm{dm}{ac_suffix}.out"
    
    return log_dir, log_filename


def set_environment_variables(config):
    """Set environment variables for the configuration."""
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
        'GPUS_PER_NODE_IRREGULAR': str(config['gpus_per_node_irregular']),
        'ACTIVATION_CHECKPOINTING': str(config['activation_checkpointing']),
        'EXIT_INTERVAL': str(config.get('exit_interval', 5)),
    }

    for key, value in env_vars.items():
        os.environ[key] = value

    # Set profiler environment variables (default: disabled)
    profiler_config = config.get('profiler', {})
    profiler_env_vars = set_profiler_environment_variables(profiler_config)
    env_vars.update(profiler_env_vars)

    return env_vars


def set_profiler_environment_variables(profiler_config: dict) -> dict:
    """
    Set environment variables for PyTorch profiler configuration.
    
    Args:
        profiler_config: Dictionary with profiler settings:
            - enabled: bool (default: False)
            - trace_dir: str (default: '', will use log_dir in salloc mode)
            - trace_prefix: str (default: '', will use log_filename in salloc mode)
            Note: schedule params (wait/warmup/active/repeat) are hardcoded in training.py
    
    Returns:
        Dictionary of environment variables set
    """
    # Default profiler config (disabled by default)
    # trace_dir and trace_prefix are overridden in run_salloc with log_dir/log_filename
    defaults = {
        'enabled': False,
        'trace_dir': '',
        'trace_prefix': '',
    }
    
    # Merge with provided config
    cfg = {**defaults, **profiler_config}
    
    env_vars = {
        'PROFILER_ENABLED': '1' if cfg['enabled'] else '0',
        'PROFILER_TRACE_DIR': str(cfg['trace_dir']) if cfg['trace_dir'] else '',
        'PROFILER_TRACE_PREFIX': str(cfg['trace_prefix']) if cfg['trace_prefix'] else '',
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
    
    return env_vars


def get_cluster_config(cluster: str) -> dict:
    """Get cluster-specific configuration."""
    if cluster not in CLUSTER_CONFIG:
        raise ValueError(f"Unknown cluster: {cluster}. Must be one of: {list(CLUSTER_CONFIG.keys())}")
    return CLUSTER_CONFIG[cluster]


def run_sbatch(config):
    """Run training via sbatch submission."""
    cluster = config.get('cluster', 'delta')
    cluster_cfg = get_cluster_config(cluster)
    
    set_environment_variables(config)
    print(f"Cluster: {cluster}")
    print("Environment variables set:")
    for key in ['TP', 'PP', 'DP', 'SP', 'GLOBAL_BATCH', 'ZERO_STAGE', 'SIZE_TAG', 'GPUS_PER_NODE_IRREGULAR']:
        print(f"  {key}={os.environ.get(key, '')}")
    
    slurm_script = cluster_cfg['slurm_script']
    cmd = ['sbatch', slurm_script]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print(result.stdout)


def run_salloc(config):
    """Run training directly in a salloc environment."""
    cluster = config.get('cluster', 'delta')
    cluster_cfg = get_cluster_config(cluster)
    
    nodes_spec = config.get('nodes')
    if not nodes_spec:
        raise ValueError("'nodes' must be specified for salloc mode (e.g., 'gpua[020,046,051,089]' for delta, 'k003-001,k003-002' for amd)")
    
    # Parse node names
    nodes = parse_slurm_nodes(nodes_spec)
    
    # Generate hostfile
    hostfile_path = generate_hostfile(
        nodes=nodes,
        gpus_per_node_irregular=config['gpus_per_node_irregular'],
        output_dir="Megatron-DeepSpeed/hostfiles"
    )
    
    # Set environment variables
    set_environment_variables(config)
    
    # Set additional environment variables for salloc mode
    master_addr = nodes[0]  # First node is master
    master_port = config.get('master_port', 13701)
    num_nodes = len(nodes)
    
    os.environ['HOSTFILE_PATH'] = os.path.abspath(hostfile_path)
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)
    os.environ['SLURM_NNODES'] = str(num_nodes)
    os.environ['CLUSTER'] = cluster  # Pass cluster to shell script
    
    # Set log directory and filename (use absolute path since script will cd to Megatron-DeepSpeed)
    log_base_dir = config.get('log_base_dir', 'logs/xxx')
    log_dir, log_filename = generate_log_path(config, num_nodes, base_dir=log_base_dir)
    log_dir_abs = os.path.abspath(log_dir)
    log_path = os.path.join(log_dir_abs, log_filename)
    
    os.environ['LOG_DIR'] = log_dir_abs
    os.environ['LOG_FILENAME'] = log_filename
    
    # Override profiler defaults based on log settings
    profiler_config = config.get('profiler', {})
    profiler_enabled = profiler_config.get('enabled', False)
    # Use log_dir_abs as default trace_dir (can be overridden by explicit trace_dir in config)
    profiler_trace_dir = profiler_config.get('trace_dir') or log_dir_abs
    os.environ['PROFILER_TRACE_DIR'] = profiler_trace_dir
    # Use log_filename (without .out) as default trace_prefix
    default_trace_prefix = log_filename.replace('.out', '') if log_filename.endswith('.out') else log_filename
    profiler_trace_prefix = profiler_config.get('trace_prefix') or default_trace_prefix
    os.environ['PROFILER_TRACE_PREFIX'] = profiler_trace_prefix
    
    print("=" * 60)
    print(f"Salloc Mode Configuration (Cluster: {cluster}):")
    print(f"  Nodes: {nodes}")
    print(f"  MASTER_ADDR: {master_addr}")
    print(f"  MASTER_PORT: {master_port}")
    print(f"  SLURM_NNODES: {num_nodes}")
    print(f"  HOSTFILE_PATH: {os.path.abspath(hostfile_path)}")
    print(f"  GPUS_PER_NODE_IRREGULAR: {config['gpus_per_node_irregular']}")
    print(f"  LOG_PATH: {log_path}")
    print(f"  PROFILER_ENABLED: {profiler_enabled}")
    if profiler_enabled:
        print(f"  PROFILER_TRACE_DIR: {profiler_trace_dir}")
        print(f"  PROFILER_TRACE_PREFIX: {profiler_trace_prefix}")
        print(f"  PROFILER_SCHEDULE: hardcoded in training.py (wait=1, warmup=3, active=1, repeat=1)")
    print("=" * 60)
    
    # Run the salloc version of the script (cluster-agnostic)
    salloc_script = cluster_cfg['salloc_script']
    cmd = ['bash', salloc_script]
    print(f"Running: {' '.join(cmd)}")
    
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


if __name__ == "__main__":
    # Example config for sbatch mode on Delta:
    # config = {
    #     'cluster': 'delta',
    #     'mode': 'sbatch',
    #     'tensor_parallel': 1,
    #     'pipeline_parallel': 1,
    #     'data_parallel': 16,
    #     'sequence_parallel': 1,
    #     'global_batch_size': 64,
    #     'grad_accum_steps': 1,
    #     'zero_stage': 3,
    #     'model_size': '7B',
    #     'num_nodes': 4,
    #     'gpus_per_node': '',
    #     'gpus_per_node_irregular': "4,4,4,4",
    #     'activation_checkpointing': 0,
    # }
    
    # Example config for sbatch mode on AMD:
    # config = {
    #     'cluster': 'amd',
    #     'mode': 'sbatch',
    #     'tensor_parallel': 4,
    #     'pipeline_parallel': 1,
    #     'data_parallel': 8,
    #     'sequence_parallel': 1,
    #     'global_batch_size': 64,
    #     'grad_accum_steps': 1,
    #     'zero_stage': 1,
    #     'model_size': '7B',
    #     'num_nodes': 4,
    #     'gpus_per_node': '',
    #     'gpus_per_node_irregular': "8,8,8,8",
    #     'activation_checkpointing': 0,
    # }
    
    # Example config for salloc mode on Delta:
    config = {
        'cluster': 'delta',  # 'delta' or 'amd'
        'mode': 'salloc',
        'nodes': 'gpua[017,038,040,044]',  # Slurm format ,040,044
        'tensor_parallel': 1,
        'pipeline_parallel': 1,
        'data_parallel': 16,
        'sequence_parallel': 1,
        'global_batch_size': 64,
        'grad_accum_steps': 1,
        'zero_stage': 3,
        'model_size': '7B',
        'num_nodes': 4,
        'gpus_per_node': '',
        'gpus_per_node_irregular': "4,4,4,4",
        'activation_checkpointing': 1,
        'master_port': 13703,
        'log_base_dir': 'logs/xxx',
        'exit_interval': 10,
        # Profiler config (optional, disabled by default)
        # trace_dir defaults to log_dir, trace_prefix defaults to log_filename
        # Note: schedule params (wait/warmup/active/repeat) are hardcoded in training.py
        'profiler': {
            'enabled': False,
            # 'trace_dir': '',  # Optional: defaults to log_dir_abs
            # 'trace_prefix': '',  # Optional: defaults to log_filename
        },
    }
    
    # Example config for salloc mode on AMD:
    # config = {
    #     'cluster': 'amd',
    #     'mode': 'salloc',
    #     'nodes': 'k003-[001-002]',  # AMD node format
    #     'tensor_parallel': 1,
    #     'pipeline_parallel': 1,
    #     'data_parallel': 8,
    #     'sequence_parallel': 1,
    #     'global_batch_size': 64,
    #     'grad_accum_steps': 1,
    #     'zero_stage': 3,
    #     'model_size': '1B',
    #     'num_nodes': 2,
    #     'gpus_per_node': '',
    #     'gpus_per_node_irregular': "4,4",
    #     'activation_checkpointing': 1,
    #     'master_port': 13701,
    #     'log_base_dir': 'logs/xxx',
    #     'exit_interval': 10,
    #     # Profiler config (optional, disabled by default)
    #     # trace_dir defaults to log_dir, trace_prefix defaults to log_filename
    #     # Note: schedule params (wait/warmup/active/repeat) are hardcoded in training.py
    #     'profiler': {
    #         'enabled': True,
    #         # 'trace_dir': '',  # Optional: defaults to log_dir_abs
    #         # 'trace_prefix': '',  # Optional: defaults to log_filename (without .out)
    #     },
    # }
    
    mode = config.get('mode', 'sbatch')
    
    if mode == 'sbatch':
        run_sbatch(config)
    elif mode == 'salloc':
        run_salloc(config)
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'sbatch' or 'salloc'")
