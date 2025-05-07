import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Nome do modelo base e checkpoint LoRA
MODEL_NAME = "deepseek-ai/deepseek-llm-1.3b-base"
CHECKPOINT_PATH = "fortesia_lora_checkpoints"

# Dispositivo de execução
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Carrega tokenizer e modelo base
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)

# Aplica LoRA no modelo base
model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
model = model.merge_and_unload()
model.eval()

# Função de geração de resposta com contexto opcional
def gerar_resposta(prompt, contexto=""):
    entrada = f"<|user|> {prompt}\n<|assistant|>"
    if contexto:
        entrada = contexto + "\n\n" + entrada
    inputs = tokenizer(entrada, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        saida = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(saida[0], skip_special_tokens=True).split("<|assistant|>")[-1].strip()

# Exemplo de uso
if __name__ == "__main__":
    print(gerar_resposta("Explique recursão em Python"))