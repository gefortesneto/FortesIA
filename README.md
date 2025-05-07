# FortesIA 🤖🌐

FortesIA é uma inteligência artificial híbrida e adaptativa que combina:

- 🔄 Roteamento inteligente de prompts para modelos especializados
- 🧠 Inferência local com LoRA (Low-Rank Adaptation)
- 🌐 Fallback para modelos externos como GPT-4, Gemini, DeepSeek
- 📚 Memória simbólica vetorial com FAISS
- 📊 Feedback contínuo via interface web
- 🐳 Deploy completo com Docker + Docker Compose

---

## 📦 Componentes

| Arquivo                          | Descrição                                      |
|----------------------------------|------------------------------------------------|
| `fortesia_cluster_balancer.py`  | Balanceador que roteia tarefas entre IAs      |
| `fortesia_lora_inferencia.py`   | Inferência local com LoRA                     |
| `fortesia_frontend_ui.html`     | Interface web leve e responsiva               |
| `fortesia_router_main.py`       | Backend FastAPI com LoRA, GPT-4, FAISS        |
| `fortesia_embed_index.py`       | Gera `.index` FAISS e `metadata.json`         |
| `fortesia_trainer.py`           | Treinamento/refinamento com LoRA              |
| `Dockerfile`                    | Backend FastAPI containerizado                |
| `docker-compose.yml`            | Orquestra backend + volumes                   |
| `requirements.txt`              | Lista de dependências                         |

---

## 🚀 Como usar (modo local)

```bash
# Clonar repositório
git clone https://github.com/gefortesneto/FortesIA.git
cd FortesIA

# Geração do index vetorial
python fortesia_embed_index.py

# Rodar local com Uvicorn
uvicorn fortesia_router_main:app --reload
```

## 🐳 Deploy com Docker

```bash
# Construir e subir containers
docker-compose up --build
```

---

## 🧪 Treinar modelo com LoRA

Certifique-se de ter um dataset `.jsonl` no formato:

```json
{ "prompt": "pergunta?", "response": "resposta." }
```

Depois execute:

```bash
python fortesia_trainer.py
```

---

## 🛠️ Tecnologias
- FastAPI
- Uvicorn
- LoRA / PEFT
- FAISS
- Sentence Transformers
- Torch / Transformers
- Docker / Compose

---

## 📄 Licença
MIT — © 2024 FortesIA Core Initiative

---

Desenvolvido com ❤️ por [@gefortesneto](https://github.com/gefortesneto)