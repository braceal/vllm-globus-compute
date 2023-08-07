import os
from vllm import LLM, SamplingParams

# TODO: Change to your directory or comment out to use home anyways
# Relocate huggingface cache dir to avoid running out of same in home
os.environ["HF_HOME"] = "/lus/eagle/projects/CVD-Mol-AI/braceal/cache/huggingface"

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
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
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

