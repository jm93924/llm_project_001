import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


base_model = "meta-llama/Llama-3.2-3B"
adapter_path = "./llama32-3b-koalpaca-lora"

tokenizer = AutoTokenizer.from_pretrained(adapter_path)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
)

model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

while True:
    instruction = input("\nInstruction: ")
    user_input = input("Input: ")

    if instruction.lower() in ["exit", "quit", "q"]:
        break

    prompt = f"""### Instruction:
{instruction}

### Input:
{user_input}

### Response:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(answer.split("### Response:")[-1].strip())