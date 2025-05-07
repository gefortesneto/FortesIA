import random
import time
import threading
from queue import Queue

# Dicionário com as URLs dos workers especializados
WORKERS = {
    "codigo": ["http://localhost:8000"],
    "imagem": ["http://localhost:8001"],
    "texto":  ["http://localhost:8002"],
    "video":  ["http://localhost:8003"]
}

# Fila de prompts
fila_tarefas = Queue()

# Classifica o tipo da tarefa com base no conteúdo do prompt
def classificar_tarefa(prompt):
    prompt = prompt.lower()
    if any(x in prompt for x in ["python", "algoritmo", "código"]):
        return "codigo"
    elif any(x in prompt for x in ["imagem", "foto", "desenho"]):
        return "imagem"
    elif any(x in prompt for x in ["vídeo", "trailer", "cena"]):
        return "video"
    return "texto"

# Thread worker que processa a fila
def executor():
    while True:
        prompt = fila_tarefas.get()
        categoria = classificar_tarefa(prompt)
        destino = random.choice(WORKERS.get(categoria, WORKERS["texto"]))
        print(f"[Balancer] Roteando: {prompt[:60]} → {destino}")
        time.sleep(1)
        fila_tarefas.task_done()

# Inicia múltiplos threads
for _ in range(4):
    threading.Thread(target=executor, daemon=True).start()

# Exemplo de uso
if __name__ == "__main__":
    prompts = [
        "Código em Python para somar dois números",
        "Imagem de um dragão azul",
        "Vídeo explicando IA",
        "O que é inteligência artificial?"
    ]
    for p in prompts:
        fila_tarefas.put(p)
    fila_tarefas.join()