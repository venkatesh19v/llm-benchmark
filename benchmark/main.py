from config import MODEL_CONFIG, PROMPT_PATH
from runner import load_model, run_generation
from utils import get_gpu_memory, get_cpu_memory
import pandas as pd
import matplotlib.pyplot as plt
import wandb

def load_prompts(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]

results = []

for model_id, model_name in MODEL_CONFIG.items():
    print(f"\nBenchmarking model: {model_id}")
    wandb.init(
        project="llm-benchmarking",
        name=f"benchmark-{model_id}",
        config={
            "model": model_id,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_new_tokens": 100,
        }
    )
    (tokenizer, model), load_time = load_model(model_name)
    print(f"Loaded in {load_time:.2f} sec")

    prompts = load_prompts(PROMPT_PATH)
    temperature = 0.7
    top_p = 0.9
    max_new_tokens = 100

    for prompt in prompts:
        try:
            output_text, tokens, gen_time = run_generation(
                tokenizer, model, prompt,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens
            )
            tps = tokens / gen_time if gen_time > 0 else 0
            words = len(output_text.split())

            result = {
                "model": model_id,
                "prompt": prompt,
                "response": output_text,
                "word_count": len(output_text.strip().split()),
                "load_time_sec": round(load_time, 2),
                "gen_time_sec": round(gen_time, 2),
                "tokens_generated": tokens,
                "temperature": temperature,
                "time_taken_sec": gen_time,
                "tokens_per_sec": round(tps, 2),
                "word_count": words,
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens,
                "cpu_mem_percent": get_cpu_memory(),
                "gpu_mem_gb": round(get_gpu_memory(), 2)
            }
            results.append(result)
            wandb.log({
                 "model": model_id,
                 "prompt": prompt,
                 "response": output_text,
                 "word_count": len(output_text.strip().split()),
                 "load_time_sec": round(load_time, 2),
                 "gen_time_sec": round(gen_time, 2),
                 "tokens_generated": tokens,
                 "temperature": temperature,
                 "time_taken_sec": gen_time,
                 "tokens_per_sec": round(tps, 2),
                 "word_count": words,
                 "temperature": temperature,
                 "top_p": top_p,
                 "max_new_tokens": max_new_tokens,
                 "cpu_mem_percent": get_cpu_memory(),
                 "gpu_mem_gb": round(get_gpu_memory(), 2)
	    })

        except Exception as e:
            print(f"Prompt failed, Error: {e}")
if results:
    df = pd.DataFrame(results)
    wandb_table = wandb.Table(dataframe=df)
    wandb.log({"benchmark_results": wandb_table})
    df.to_csv("results/benchmark_report.csv", index=False)
    print("Benchmark saved to results/benchmark_report.csv")
else:
    print("No results to save, benchmark skipped.")
wandb.finish()
