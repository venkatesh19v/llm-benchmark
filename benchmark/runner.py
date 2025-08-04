import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import get_gpu_memory, get_cpu_memory, timing
import time

@timing
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return tokenizer, model

def run_generation(tokenizer, model, prompt, temperature=0.7, top_p=0.9, max_new_tokens=100):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    start_time = time.time()
    outputs = model.generate(
        **inputs,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id  # Avoid warning
    )
    gen_time = time.time() - start_time

    generated_tokens = outputs.shape[-1] - inputs.input_ids.shape[-1]
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return output_text, generated_tokens, gen_time 
