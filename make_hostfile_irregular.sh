#!/bin/bash
# This script is used to generate a hostfile for irregular GPU configuration.
# It is used in the pretrain_gpt_on_delta_torchrun_irregular_mesh.slurm script.
# It reads the GPUS_PER_NODE_IRREGULAR environment variable
# The GPUS_PER_NODE_IRREGULAR environment variable is set in the pretrain_gpt_on_delta_torchrun_irregular_mesh.slurm script.
# The GPUS_PER_NODE_IRREGULAR environment variable is a comma-separated list of the number of GPUs per node.
set -euo pipefail

function makehostfile() {
perl -e '
# get node list (in scontrol order)
my @nodes = split /\n/, qx[scontrol show hostnames $ENV{"SLURM_JOB_NODELIST"}];
die "No nodes found from SLURM_JOB_NODELIST\n" unless @nodes;

# try to read irregular GPU configuration
my $irreg = $ENV{"GPUS_PER_NODE_IRREGULAR"} // "";
$irreg =~ s/\s+//g;   # remove whitespace

my @gpn_list = ();

if ($irreg ne "") {
    # GPUS_PER_NODE_IRREGULAR is provided, validate it
    @gpn_list = split /,/, $irreg;

    # validate length consistency
    if (scalar(@gpn_list) != scalar(@nodes)) {
        die sprintf(
            "GPUS_PER_NODE_IRREGULAR length (%d) must equal number of nodes (%d)\n",
            scalar(@gpn_list), scalar(@nodes)
        );
    }

    # validate all are positive integers
    for my $i (0..$#gpn_list) {
        my $v = $gpn_list[$i];
        die "Non-positive or non-integer GPU count at index $i: [$v]\n"
            unless defined($v) && $v =~ /^\d+$/ && $v > 0;
    }
} else {
    # GPUS_PER_NODE_IRREGULAR is empty, fallback to GPUS_PER_NODE
    my $default_gpn = $ENV{"GPUS_PER_NODE"};
    die "GPUS_PER_NODE environment variable must be set when GPUS_PER_NODE_IRREGULAR is empty\n" 
        if !defined($default_gpn) || $default_gpn eq "";
    
    # fill gpn_list with default value for all nodes
    @gpn_list = ($default_gpn) x scalar(@nodes);
}

# output hostfile
for my $i (0..$#nodes) {
    my $slots = $gpn_list[$i];
    print "$nodes[$i] slots=$slots\n";
}
'
}

# ensure directory exists
mkdir -p Megatron-DeepSpeed/hostfiles

# generate hostfile
makehostfile > "Megatron-DeepSpeed/hostfiles/hostfile_${SLURM_JOB_ID}"
echo "Hostfile created: Megatron-DeepSpeed/hostfiles/hostfile_${SLURM_JOB_ID}"
