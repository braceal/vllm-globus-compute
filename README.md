# vllm-globus-compute
vLLM with Globus Compute

This demonstrates how to run a vLLM instance on Polaris and submit inference
prompts from any computer and receive the response from a model of your choice
back to the computer you submitted the task from.

**Note**: This is a hands-on repo. It will be helpful to read all the code files since you will be editing them by the `TODO` statements.
  

# Install

Please run all installation commands on an interactive node so that it uses the correct build environment:
```bash
qsub -I -l select=1 -l walltime=1:00:00 -A <project-id> -q debug -l filesystems=home:eagle
```

Build the environment:
```bash
module load conda/2023-01-10-unstable
mamba env create -f environment.yml
conda activate vllm
```

> :warning: This installation no longer works since new vllm versions require new nvidia drivers. Please build the env using `environment.yml` instead.

**Note**: xformers needs a particular version of python to avoid building from source
```bash
module load conda/2023-01-10-unstable
conda create -n vllm python=3.10.11 -y
conda activate vllm
pip install vllm
```

Before running, we recommend you put your huggingface cache directory
on a file system with a larger quota than the home directory, e.g.,
```bash
echo 'export HF_HOME=/lus/eagle/projects/CVD-Mol-AI/braceal/cache/huggingface' >> ~/.bashrc
source ~/.bashrc
```

**Important**: Please make sure to update the path in the .py files as well (sorry it's a bit hacky)

# Check that xformers is properly installed
```bash
python -m xformers.info
```

It should output:
```console
xFormers 0.0.20
memory_efficient_attention.cutlassF:               available
memory_efficient_attention.cutlassB:               available
memory_efficient_attention.flshattF:               available
memory_efficient_attention.flshattB:               available
memory_efficient_attention.smallkF:                available
memory_efficient_attention.smallkB:                available
memory_efficient_attention.tritonflashattF:        available
memory_efficient_attention.tritonflashattB:        available
indexing.scaled_index_addF:                        available
indexing.scaled_index_addB:                        available
indexing.index_select:                             available
swiglu.dual_gemm_silu:                             available
swiglu.gemm_fused_operand_sum:                     available
swiglu.fused.p.cpp:                                available
is_triton_available:                               True
is_functorch_available:                            False
pytorch.version:                                   2.0.1+cu117
pytorch.cuda:                                      available
gpu.compute_capability:                            8.0
gpu.name:                                          NVIDIA A100-SXM4-40GB
build.info:                                        available
build.cuda_version:                                1108
build.python_version:                              3.10.11
build.torch_version:                               2.0.1+cu118
build.env.TORCH_CUDA_ARCH_LIST:                    5.0+PTX 6.0 6.1 7.0 7.5 8.0 8.6
build.env.XFORMERS_BUILD_TYPE:                     Release
build.env.XFORMERS_ENABLE_DEBUG_ASSERTIONS:        None
build.env.NVCC_FLAGS:                              None
build.env.XFORMERS_PACKAGE_FROM:                   wheel-v0.0.20
build.nvcc_version:                                11.8.89
source.privacy:                                    open source
```

At this point you should be able to run
```bash
python offline_inference.py
```

# Setup a globus-compute endpoint to run LLM inference from anywhere

First, install and run the command to configure the endpoint
```bash
pip install globus-compute-endpoint
globus-compute-endpoint configure
globus-compute-endpoint configure vllm
```

The above command will output a yaml file with the globus-compute configuration.
Let's update it by copying this in. Make sure to update the **account** name so your jobs get charged correctly.

For more details, see here: https://globus-compute.readthedocs.io/en/latest/endpoints.html#polaris-alcf

Update this file: `~/.globus_compute/vllm/config.yaml` 
```yaml
engine:
    type: HighThroughputEngine
    max_workers_per_node: 1

    # Un-comment to give each worker exclusive access to a single GPU
    available_accelerators: 4

    strategy:
        type: SimpleStrategy
        max_idletime: 300

    address:
        type: address_by_interface
        ifname: bond0

    provider:
        type: PBSProProvider

        launcher:
            type: MpiExecLauncher
            # Ensures 1 manger per node, work on all 64 cores
            bind_cmd: --cpu-bind
            overrides: --depth=64 --ppn 1

        account: RL-fold
        queue: preemptable
        cpus_per_node: 32
        select_options: ngpus=4

        # e.g., "#PBS -l filesystems=home:grand:eagle\n#PBS -k doe"
        scheduler_options: "#PBS -l filesystems=home:grand:eagle"

        # Node setup: activate necessary conda environment and such
        worker_init: "module load conda/2023-01-10-unstable; conda activate vllm"

        walltime: 01:00:00
        nodes_per_block: 1
        init_blocks: 0
        min_blocks: 0
        max_blocks: 2
```

Finally, we just need to start the globus-compute endpoint so it can begin
receiving requests. Run this and follow the prompt:
```bash
globus-compute-endpoint start vllm
```

You should see that the `vllm` endpoint is in the `Running` state:
```bash
globus-compute-endpoint list
```

**Important**: Take the UUID next to your `vllm` endpoint and copy it into `offline_inference_gc.py` under the `TODO`.

You can stop your endpoint by running:
```bash
globus-compute-endpoint stop vllm
```

# Running the LLM inference from anywhere
Now that the globus-compute endpoint is running, we can open a new terminal and
run the vllm inference function locally and receive the response back from the
model running on Polaris.

**Note**: We need to use the same python version as is running on Polaris.

**Note**: The first time your run the function on a new computer you will need to authenticate with Globus.

**Note**: If it's been a while since you've run the function, it will take a minute to warm up. That's because the job has to get through the queue and import the python libraries, etc. Subsequent calls will be faster.

Locally (or from any computer),
```bash
conda create -n vllm python=3.10.11 -y
conda activate vllm
pip install globus-compute-sdk
python offline_inference_gc.py
```

# Next steps
Now you are able to run inference from anywhere. Try playing with different models, sampling parameters, prompts, etc. Build a CLI to experiment, call your function from other software, have fun!

