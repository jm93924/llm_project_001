import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel
from trl import SFTTrainer, SFTConfig


# =========================
# 1. 기본 모델 및 Phase 1 어댑터 경로
# =========================

base_model = "meta-llama/Llama-3.2-3B"

# 이미 Alpaca로 학습한 Phase 1 LoRA 어댑터
phase1_adapter_path = "./llama32-3b-alpaca-lora"

# Phase 2 학습 결과를 저장할 새 폴더
output_adapter_path = "./llama32-3b-auto-drive-lora"


# =========================
# 2. 토크나이저 로드
# =========================

tokenizer = AutoTokenizer.from_pretrained(phase1_adapter_path)
tokenizer.pad_token = tokenizer.eos_token


# =========================
# 3. 4bit QLoRA 설정
# =========================
# 4090에서 bf16이 정상 지원되면 torch.bfloat16 가능
# 만약 bf16 오류가 나면 torch.float16으로 바꾸고,
# 아래 SFTConfig도 bf16=False, fp16=True로 변경

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


# =========================
# 4. Base model 로드
# =========================

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
)


# =========================
# 5. Phase 1 Alpaca LoRA 어댑터 로드
# =========================
# 핵심:
# 기존 Alpaca instruction tuning 능력을 가진 어댑터를 불러오고,
# is_trainable=True로 설정해서 이 어댑터를 이어서 추가 학습함.

model = PeftModel.from_pretrained(
    model,
    phase1_adapter_path,
    is_trainable=True,
)


# =========================
# 6. 데이터셋 로드
# =========================
# risk: 직접 만든 자율주행 위험도 데이터
# alpaca_replay: 기존 대화 능력 유지를 위한 Alpaca 일부 replay 데이터
#
# 예: risk 100개 + alpaca 25개 = 대략 80:20 비율

risk = load_dataset(
    "json",
    data_files="data/auto-drive-data.json",
    split="train",
)

alpaca_replay = load_dataset(
    "tatsu-lab/alpaca",
    split="train[:25]",
)

dataset = concatenate_datasets([risk, alpaca_replay])
dataset = dataset.shuffle(seed=42)


# =========================
# 7. Alpaca 형식 포맷팅 함수
# =========================
# input이 있으면 Instruction + Input + Response
# input이 없으면 Instruction + Response

def formatting_func(example):
    if example["input"]:
        return f"""### Instruction:
{example["instruction"]}

### Input:
{example["input"]}

### Response:
{example["output"]}"""
    else:
        return f"""### Instruction:
{example["instruction"]}

### Response:
{example["output"]}"""


# =========================
# 8. Phase 2 학습 설정
# =========================
# Phase 2는 이미 학습된 adapter를 미세 조정하는 단계라
# learning_rate를 낮게 잡는 것이 안전함.
#
# 네 환경에서는 max_seq_length가 아니라 max_length 사용.

training_args = SFTConfig(
    output_dir="./results_phase2",

    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,

    num_train_epochs=2,
    learning_rate=5e-6,

    max_length=2048,

    bf16=True,
    fp16=False,

    save_strategy="epoch",
    save_total_limit=2,

    logging_dir="./logs_phase2",
    logging_strategy="steps",
    logging_steps=10,
    logging_first_step=True,

    report_to=["tensorboard"],
    run_name="llama32_phase2_autonomous_risk",
)


# =========================
# 9. SFTTrainer 생성
# =========================
# peft_config는 넣지 않음.
# 이유:
# 이미 Phase 1 LoRA 어댑터를 PeftModel로 불러왔기 때문.
# 여기서 새 peft_config를 또 넣으면 새 LoRA를 만들려 하거나 꼬일 수 있음.

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    formatting_func=formatting_func,
    args=training_args,
)


# =========================
# 10. 학습 실행
# =========================

trainer.train()


# =========================
# 11. Phase 2 어댑터 저장
# =========================
# 기존 Alpaca 어댑터를 덮어쓰지 않고 새 경로에 저장

trainer.save_model(output_adapter_path)
tokenizer.save_pretrained(output_adapter_path)

print(f"Phase 2 adapter saved to: {output_adapter_path}")