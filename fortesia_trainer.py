from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
from transformers import Trainer

MODEL_NAME = "deepseek-ai/deepseek-llm-1.3b-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.05, bias="none")
model = get_peft_model(model, config)

# Dataset de treino no formato JSONL
dataset = load_dataset("json", data_files="oasst_pairs.jsonl")['train']

def tokenize(batch):
    return tokenizer(batch['prompt'] + "\n" + batch['response'], truncation=True, padding="max_length", max_length=512)

dataset = dataset.map(tokenize)

args = TrainingArguments(
    output_dir="fortesia_lora_checkpoints",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    save_steps=100,
    logging_steps=10,
    report_to="none"
)

trainer = Trainer(model=model, args=args, train_dataset=dataset)
trainer.train()
model.save_pretrained("fortesia_lora_checkpoints")