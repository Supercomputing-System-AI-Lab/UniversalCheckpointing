import argparse
import os
import re
from collections import defaultdict
import concurrent.futures
import math

import torch
from deepspeed.checkpoint import DeepSpeedCheckpoint
from deepspeed.checkpoint import (
    PARAM,
    CAT_DIM,
    PARAM_N_SUB_PARAMS,
    SUB_PARAM_SHAPE,
    PARAM_SHAPES,
    VOCAB_TENSOR,
    UNIVERSAL_CHECKPOINT_INFO,
    VOCABULARY_PARAMETER_PATTERNS,
    TP_REPLICATED_PARAMETER_PATTERNS,
    PARAMETER_TO_AVERAGE_PATTERNS,
    PARAMETER_WITH_ROW_PARALLELISM_PATTERNS,
    PARAMETER_WITH_2_SUB_PARAMS_CAT_DIM_0,
    PARAMETER_WITH_SUB_PARAMS,
    SubparamShape,
)

from timing import Timer

device='cpu'
assert device=='cpu'

# Thread-local checkpoint cache
# Each worker will maintain its own cache
import threading
local_cache = threading.local()

def extract_ranks_from_filename(filename, tp_degree):
    pattern = r"bf16_zero_pp_rank_(\d+)_mp_rank_(\d+)_optim_states\.pt"
    match = re.match(pattern, filename)
    if match:
        dp_rank = int(match.group(1))
        mp_rank = int(match.group(2))
        pp_rank = mp_rank // tp_degree
        tp_rank = mp_rank % tp_degree
        return dp_rank, pp_rank, tp_rank
    return None, None, None

def dist_ckpt_key_mapping(input_folder, tp_degree, sanity_check=False):
    # complexity: O(FxK). F: number of files, K: average number of keys per file
    key_to_files = defaultdict(list)  # param_name -> list of (file, group_id, slice_mapping, (3dranks))
    group_keys = defaultdict(dict)  # (dp_rank, pp_rank) -> keys set
    param_orders_by_ranks = defaultdict(list)  # (pp_rank, dp_rank, tp_rank) -> list of param_names

    file_ranks = {}  # fname -> (dp_rank, pp_rank)
    all_keys_by_file = {}

    for fname in os.listdir(input_folder):
        if not fname.endswith(".pt"):
            continue
        dp_rank, pp_rank, tp_rank = extract_ranks_from_filename(fname, tp_degree)
        if dp_rank is None:
            continue
        path = os.path.join(input_folder, fname)
        try:
            ckpt = torch.load(path, map_location=device, weights_only=False, mmap=True)
        except Exception as e:
            print(f"Failed to load {fname}: {e}")
            continue

        keys = set()
        for group_id, group in enumerate(ckpt['optimizer_state_dict']['param_slice_mappings']):
            keys.update(group.keys())
            # Mapping param_name -> (file, group_id, slice_mapping, (3dranks))
            for key in group.keys():
                key_to_files[key].append((fname, group_id, group[key], (dp_rank, pp_rank, tp_rank)))
            param_orders_by_ranks[(pp_rank, dp_rank, tp_rank)].extend(group.keys())
        all_keys_by_file[fname] = keys
        file_ranks[fname] = (dp_rank, pp_rank)

        if sanity_check:
            # sanity check: tp group consistency
            key_sig = group_keys.get((dp_rank, pp_rank))
            if key_sig is None:
                group_keys[(dp_rank, pp_rank)] = keys
            elif key_sig != keys:
                print(f"x Key mismatch in the tp group (dp={dp_rank}, pp={pp_rank})")
                print(f"File: {fname}")
                print(f"Expected keys: {sorted(key_sig)}")
                print(f"Found keys: {sorted(keys)}")
        else:
            group_keys[(dp_rank, pp_rank)] = keys

    if sanity_check:
        # sanity check: disjoint keys in the pp group
        pp_to_keys = defaultdict(set)
        for fname, keys in all_keys_by_file.items():
            _, pp_rank = file_ranks[fname]
            pp_to_keys[pp_rank].update(keys)

        pp_ranks = list(pp_to_keys.keys())
        for i in range(len(pp_ranks)):
            for j in range(i + 1, len(pp_ranks)):
                pp1, pp2 = pp_ranks[i], pp_ranks[j]
                if pp_to_keys[pp1] & pp_to_keys[pp2]:
                    print(f"x Key overlap between pp_rank={pp1} and pp_rank={pp2}")

    # get param iterate order from the param_orders_by_ranks
    param_iterate_order = [p for _, v in sorted(param_orders_by_ranks.items()) for p in v]
    return key_to_files, param_iterate_order

def get_checkpoint(filepath):
    """Thread-local function to get checkpoint from cache or load it"""
    # Initialize cache for this thread if it doesn't exist
    if not hasattr(local_cache, 'ckpt_cache'):
        local_cache.ckpt_cache = {}
        local_cache.access_order = []
    
    if filepath not in local_cache.ckpt_cache:
        local_cache.ckpt_cache[filepath] = torch.load(filepath, map_location=device, mmap=True, weights_only=False)
        local_cache.access_order.append(filepath)
        
        # Keep cache size under control - max 8 files per thread
        if len(local_cache.ckpt_cache) > 8:
            # Remove oldest accessed file
            oldest_key = local_cache.access_order.pop(0)
            del local_cache.ckpt_cache[oldest_key]
    else:
        # Move this file to the end of access_order to mark it as recently used
        local_cache.access_order.remove(filepath)
        local_cache.access_order.append(filepath)
                
    return local_cache.ckpt_cache[filepath]

def process_param_batch(batch_data):
    """
    Process a batch of adjacent parameters
    Each batch is handled by a single worker to maximize cache locality
    """
    param_data_list, batch_id, total_batches = batch_data
    params_processed = 0
    batch_size = len(param_data_list)
    results = []
    
    for param_data in param_data_list:
        (param_name, file_list, shape, input_folder, zero_output_folder, 
         replicated_parameters, parameters_to_average, parameters_with_row_parallelism,
         vocabulary_parameters, parameters_with_2_sub_params_cat_dim_0, 
         parameter_with_sub_params, universal_checkpoint_info) = param_data
        
        param_base_path = os.path.join(zero_output_folder, param_name)
        os.makedirs(param_base_path, exist_ok=True)

        spanned_tp_ranks = sorted(set(x[3][2] for x in file_list))
        spanned_dp_ranks = sorted(set(x[3][0] for x in file_list))
        
        def get_matched_pattern(patterns_, name_):
            matched_ = [pattern_ for pattern_ in patterns_ if re.match(pattern_, name_)]
            assert len(matched_) <= 1, f'Got more than one matching patterns={matched_} for {name_}'
            if matched_:
                return matched_[0]
            return None

        def get_matched_sub_params_pattern(name_):
            for subparam_shape_dict in parameter_with_sub_params:
                subparam_shape = SubparamShape(**subparam_shape_dict)
                for pattern_ in subparam_shape.patterns:
                    if re.match(pattern_, name_):
                        return subparam_shape
            return None

        for state in ("fp32", "exp_avg", "exp_avg_sq"):
            tp_shards = []

            for tp_index in spanned_tp_ranks:
                dp_shards = []
                for dp_index, dp_rank in enumerate(spanned_dp_ranks):
                    file_index = tp_index * len(spanned_dp_ranks) + dp_index
                    if file_index >= len(file_list):
                        raise IndexError(f"Insufficient files for param={param_name}, tp_index={tp_index}, dp_index={dp_index}")

                    file, group_id, slice_mapping, _ = file_list[file_index]
                    start, numel = slice_mapping.start, slice_mapping.numel
                    
                    # Get checkpoint from thread-local cache
                    ckpt_path = os.path.join(input_folder, file)
                    ckpt = get_checkpoint(ckpt_path)

                    if state == "fp32":
                        dp_shard = ckpt['optimizer_state_dict']['single_partition_of_fp32_groups'][group_id][start:start+numel]
                    else:
                        # "exp_avg" or "exp_avg_sq"
                        dp_shard = ckpt['optimizer_state_dict']['base_optimizer_state']['state'][group_id][state][start:start+numel]

                    dp_shards.append(dp_shard)

                if shape is None:
                    tp_shard = torch.cat(dp_shards, dim=0)
                else:
                    try:
                        tp_shard = torch.cat(dp_shards, dim=0).reshape(shape)
                    except Exception as e:
                        print(f"Error reshaping {param_name} to shape {shape}")
                        raise e

                tp_shards.append(tp_shard)

            ckpt_dict = {}
            matched_sub_params_shape = get_matched_sub_params_pattern(param_name)
            
            if get_matched_pattern(replicated_parameters, param_name):
                # if len(tp_shards) > 1:
                #     assert all([tp_shards[0].equal(other_slice) for other_slice in tp_shards[1:]])
                full_param = tp_shards[0]
            elif get_matched_pattern(parameters_to_average, param_name):
                full_param = sum(tp_shards) / len(tp_shards)
            elif get_matched_pattern(parameters_with_2_sub_params_cat_dim_0, param_name):
                cat_dim = 0
                chunked_slices = [torch.chunk(s, 2, dim=cat_dim) for s in tp_shards]
                merged_chunks_0 = torch.cat([s[0] for s in chunked_slices], dim=cat_dim)
                merged_chunks_1 = torch.cat([s[1] for s in chunked_slices], dim=cat_dim)
                full_param = torch.cat([merged_chunks_0, merged_chunks_1], dim=cat_dim)
                ckpt_dict[CAT_DIM] = cat_dim
                ckpt_dict[PARAM_N_SUB_PARAMS] = 2
            elif matched_sub_params_shape:
                merged_chunks = []
                partition_dim = matched_sub_params_shape.partition_dim

                sub_dim_sizes = matched_sub_params_shape.shape[partition_dim]
                if not isinstance(sub_dim_sizes, tuple):
                    sub_dim_sizes = (sub_dim_sizes, )

                partition_shape = [sum(d) if isinstance(d, tuple) else d for d in matched_sub_params_shape.shape]
                tp_degree = len(spanned_tp_ranks)
                partition_shape = [d // tp_degree if i == partition_dim else d for i, d in enumerate(partition_shape)]
                tp_shards = [s.view(partition_shape) for s in tp_shards]

                offset = 0
                for sub_dim_size in sub_dim_sizes:
                    part_sub_dim_size = sub_dim_size // tp_degree
                    merged_chunks.append(
                        torch.cat([s.narrow(partition_dim, offset, part_sub_dim_size) for s in tp_shards], dim=partition_dim))
                    offset += part_sub_dim_size
                full_param = torch.cat(merged_chunks, dim=partition_dim)
                ckpt_dict[SUB_PARAM_SHAPE] = matched_sub_params_shape
            else:
                cat_dim = 1 if get_matched_pattern(parameters_with_row_parallelism, param_name) else 0
                full_param = torch.cat(tp_shards, dim=cat_dim)
                ckpt_dict[CAT_DIM] = cat_dim

            if get_matched_pattern(vocabulary_parameters, param_name):
                # strip padding
                original_vocab_size = universal_checkpoint_info['original_vocab_size']
                full_param = full_param[:original_vocab_size, :]
                ckpt_dict[VOCAB_TENSOR] = True

            # Save merged param
            final_path = os.path.join(param_base_path, f"{state}.pt")
            torch.save({PARAM: full_param, **ckpt_dict}, final_path)
        
        results.append(param_name)
        params_processed += 1
        
        # Report progress within batch
        if params_processed % max(1, batch_size // 5) == 0 or params_processed == batch_size:
            print(f"Batch {batch_id+1}/{total_batches}: Processed {params_processed}/{batch_size} parameters ({params_processed/batch_size*100:.1f}%)")
    
    return results

def merge_and_save_params(key_to_files, input_folder, output_folder, num_workers, slice_shapes, ds_checkpoint, param_iterate_order):
    """
    Parallelized version with batch-based parameter processing
    Groups adjacent parameters together to be processed by the same worker
    """
    zero_output_folder = os.path.join(output_folder, "zero")
    os.makedirs(zero_output_folder, exist_ok=True)

    # Get patterns from checkpoint info
    universal_checkpoint_info = ds_checkpoint.get_checkpoint_info(UNIVERSAL_CHECKPOINT_INFO)
    replicated_parameters = universal_checkpoint_info.get(TP_REPLICATED_PARAMETER_PATTERNS, [])
    parameters_to_average = universal_checkpoint_info.get(PARAMETER_TO_AVERAGE_PATTERNS, [])
    parameters_with_row_parallelism = universal_checkpoint_info.get(PARAMETER_WITH_ROW_PARALLELISM_PATTERNS, [])
    vocabulary_parameters = universal_checkpoint_info.get(VOCABULARY_PARAMETER_PATTERNS, [])
    parameters_with_2_sub_params_cat_dim_0 = universal_checkpoint_info.get(PARAMETER_WITH_2_SUB_PARAMS_CAT_DIM_0, [])
    parameter_with_sub_params = universal_checkpoint_info.get(PARAMETER_WITH_SUB_PARAMS, [])

    # Create a list of parameters to process
    saved_params = set()
    params_to_process = []
    
    for param_name in param_iterate_order:
        if param_name in saved_params:
            continue
        
        file_list = key_to_files[param_name]
        shape = slice_shapes.get(param_name, None)
        
        # Sort file_list by (tp_rank, dp_rank)
        file_list.sort(key=lambda x: (x[3][2], x[3][0]))  # (tp_rank, dp_rank)
        
        # Add parameter to the processing list
        params_to_process.append((
            param_name, 
            file_list, 
            shape, 
            input_folder,
            zero_output_folder,
            replicated_parameters,
            parameters_to_average,
            parameters_with_row_parallelism,
            vocabulary_parameters,
            parameters_with_2_sub_params_cat_dim_0,
            parameter_with_sub_params,
            universal_checkpoint_info
        ))
        
        saved_params.add(param_name)
    
    # Determine number of workers based on CPU count
    # import multiprocessing
    num_workers = num_workers #32 #max(1, multiprocessing.cpu_count() // 2)
    
    # Determine batch size based on number of parameters and workers
    total_params = len(params_to_process)
    batch_size = math.ceil(total_params / num_workers)
    
    # Create batches of parameters to process - keeping adjacent parameters together
    param_batches = []
    for i in range(0, total_params, batch_size):
        batch = params_to_process[i:min(i + batch_size, total_params)]
        param_batches.append((batch, i // batch_size, len(range(0, total_params, batch_size))))
    
    print(f"Processing {total_params} parameters in {len(param_batches)} batches using {num_workers} workers")
    print(f"Each batch contains approximately {batch_size} adjacent parameters")
    
    # Process batches in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        print(f"Multi-threading: {total_params} parameters in {len(param_batches)} batches using {num_workers} workers")
        # Submit all batches and collect futures
        futures = {executor.submit(process_param_batch, batch_data): batch_idx 
                  for batch_idx, batch_data in enumerate(param_batches)}
        
        # Process results as they complete
        completed_batches = 0
        completed_params = 0
        for future in concurrent.futures.as_completed(futures):
            batch_idx = futures[future]
            # try:
            results = future.result()
            completed_batches += 1
            completed_params += len(results)
            print(f"Completed batch {batch_idx+1}/{len(param_batches)}")
            print(f"Overall progress: {completed_params}/{total_params} parameters processed ({completed_params/total_params*100:.1f}%)")
            # except Exception as e:
            #     print(f"Error processing batch {batch_idx+1}: {e}")

def split_param_list_preserve_order(param_iterate_order, slice_shapes, world_size, rank) -> list[str]:
    if world_size == 1:
        return param_iterate_order

    # count each param's size
    name_and_param_sizes = []
    for param_name in param_iterate_order:
        name_and_param_sizes.append((param_name, slice_shapes[param_name].numel()))

    # split into chunks
    k = world_size
    total_elements = sum(elems for _, elems in name_and_param_sizes)
    target = total_elements / k

    chunks = [[] for _ in range(k)]
    current_chunk = 0
    current_sum = 0

    for name, elems in name_and_param_sizes:
        if current_chunk < k - 1 and current_sum + elems > target:
            current_chunk += 1
            current_sum = 0
        chunks[current_chunk].append(name)
        current_sum += elems

    return chunks[rank]

def test_dist_ckpt_key_mapping():
    input_folder = "path/to/dcp_folder"
    tp_degree = 8
    key_to_files, param_iterate_order = dist_ckpt_key_mapping(input_folder, tp_degree)
    # number of files -> list of keys
    num_files_to_keys = defaultdict(list)
    for k, v in key_to_files.items():
        num_files_to_keys[len(v)].append(k)
    print(num_files_to_keys.keys())
    num_files_to_keys.pop(tp_degree)
    for k, v in num_files_to_keys.items():
        print(k, v)

def test_merge_and_save_params(args):
    input_folder = args.input_folder
    output_folder = args.output_folder
    num_workers = args.num_workers
    ds_checkpoint = DeepSpeedCheckpoint(input_folder)
    tp_degree = ds_checkpoint.tp_degree

    # get slice shapes
    slice_shapes = []
    for mp_rank_file in ds_checkpoint.mp_rank_files:
        mp_sd = torch.load(mp_rank_file, map_location=torch.device(device), weights_only=False)
        slice_shapes += mp_sd[PARAM_SHAPES]
    slice_shapes = dict((k, v) for d in slice_shapes for k, v in d.items())  # fix back to normal flat dict, merge duplicates for tp>1

    with Timer("get key to files"):
        key_to_files, param_iterate_order = dist_ckpt_key_mapping(input_folder, tp_degree)

    if args.mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        world_size = comm.Get_size()
        rank = comm.Get_rank()

        if rank == 0:
            from torch_io_logger import measure_io
            torch.load = measure_io("torch.load")(torch.load)
            torch.save = measure_io("torch.save")(torch.save)

        # split across ranks if using multiple nodes
        # return the list of params to process for this rank
        param_iterate_order = split_param_list_preserve_order(param_iterate_order, slice_shapes, world_size, rank)
    else:
        comm = None
        world_size = 1
        rank = 0

    total_elements = sum(slice_shapes[param_name].numel() for param_name in param_iterate_order)
    print(f"Rank {rank} has {len(param_iterate_order)} parameters to process, total {total_elements}")

    with Timer("merge and save params (batch parallel)", comm=comm):
        merge_and_save_params(key_to_files, input_folder, output_folder, num_workers, slice_shapes, ds_checkpoint, param_iterate_order)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--mpi", action="store_true")
    args = parser.parse_args()
    test_merge_and_save_params(args)