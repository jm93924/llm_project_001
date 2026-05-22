import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


base_model = "meta-llama/Llama-3.2-3B"
adapter_path = "./llama32-3b-auto-drive-lora"

tokenizer = AutoTokenizer.from_pretrained(adapter_path)
tokenizer.pad_token = tokenizer.eos_token

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

history = []

while True:
    if len(history) > 6:
        history = history[-6:]

    user_input = input("\nUser: ")

    if user_input.lower() in ["exit", "quit", "q"]:
        break

    history.append({"role": "user", "content": user_input})

    prompt = """### System:
너는 답변하는 챗봇이다.

"""

    for msg in history:
        if msg["role"] == "user":
            prompt += f"### User:\n{msg['content']}\n\n"
        else:
            prompt += f"### Assistant:\n{msg['content']}\n\n"

    prompt += "### Assistant:\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print(f"\n[Prompt tokens]: {inputs['input_ids'].shape[1]}")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.4,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = full_text.split("### Assistant:")[-1]
    answer = answer.split("### User:")[0].strip()

    print("\nAssistant:", answer)

    history.append({"role": "assistant", "content": answer})