import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
import time

model_name = "Qwen/Qwen2.5-7B-Instruct"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

print("Loading 7B 4bit model...")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto"
)

print("Model loaded")
print("Device:", model.device)

print(
    "GPU memory allocated:",
    torch.cuda.memory_allocated()/1024**3,
    "GB"
)

prompt = "介绍一下人工智能的发展历史和未来趋势。"

inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to("cuda")


# warm up
with torch.no_grad():
    _ = model.generate(
        **inputs,
        max_new_tokens=20
    )

torch.cuda.synchronize()


torch.cuda.reset_peak_memory_stats()

start = time.time()

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100
    )

torch.cuda.synchronize()

end = time.time()


print(
    "Peak memory:",
    torch.cuda.max_memory_allocated()/1024**3,
    "GB"
)

print(
    "Time:",
    end-start
)

print(
    "Tokens/s:",
    100/(end-start)
)


print(
    tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )
)
