import time
from typing import List, Dict
from globus_compute_sdk import Executor

def _run_vllm(prompts):
    import os
    from vllm import LLM, SamplingParams

    # Hack to avoid vllm AssertionError: data parallel group is already initialized
    # caused by the parsl job caching module level globals.
    from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
    destroy_model_parallel()

    # TODO: Change to your directory or comment out to use home anyways
    # Relocate huggingface cache dir to avoid running out of same in home
    os.environ["HF_HOME"] = "/lus/eagle/projects/CVD-Mol-AI/braceal/cache/huggingface"

    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.7, top_p=0.95)

    # Create an LLM.
    #llm = LLM(model="facebook/opt-125m")
    #llm = LLM(model="openlm-research/open_llama_13b")
    llm = LLM(model="mosaicml/mpt-7b-storywriter", trust_remote_code=True)
    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    return_outputs = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        #print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        return_outputs.append({"prompt": prompt, "response": generated_text})

    return return_outputs

def run_vllm() -> List[Dict[str, str]]:
    """Client side function."""

    # TODO: The vllm endpoint UUID on Polaris
    endpoint = "5228c1c5-fc96-44d6-ad77-578281186fa6"
    
    # Initialize the compute executor
    gce = Executor(endpoint_id=endpoint)

    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    # Submit the remote procedure call
    fut = gce.submit(_run_vllm, prompts)

    print("Running function", fut)

    # Collect the return result
    result = fut.result()

    return result

if __name__ == "__main__":
    results = run_vllm()
    for result in results:
        print(f"Prompt: {result['prompt']!r}, Generated text: {result['response']!r}")

