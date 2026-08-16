import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="auto"
)

print("Model device:")
print(next(model.parameters()).device)

text = "介绍一下人工智能的发展历史"

inputs = tokenizer(
    text,
    return_tensors="pt"
).to(model.device)

with torch.no_grad():
    import time

    start = time.time()

    outputs = model.generate(
        **inputs,
        max_new_tokens=100
    )

    end = time.time()

    print("Time:", end-start)
    print("Tokens/s:", 100/(end-start))

print(
    tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )
)
