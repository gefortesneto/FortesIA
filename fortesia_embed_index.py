from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np

# Frases exemplo
exemplos = [
    {"prompt": "O que é aprendizado de máquina?", "response": "É uma subárea da inteligência artificial..."},
    {"prompt": "Explique redes neurais.", "response": "São modelos computacionais inspirados no cérebro humano..."},
    {"prompt": "Como funciona o LoRA?", "response": "LoRA é uma técnica de ajuste leve de grandes modelos..."}
]

# Gera embeddings
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
vetores = model.encode([ex["prompt"] for ex in exemplos], normalize_embeddings=True)
vetores = np.array(vetores).astype("float32")

# Cria índice FAISS
index = faiss.IndexFlatIP(vetores.shape[1])
index.add(vetores)

# Salva os arquivos
faiss.write_index(index, "fortesia_faiss.index")
with open("fortesia_metadata.json", "w", encoding="utf-8") as f:
    json.dump(exemplos, f, indent=2, ensure_ascii=False)

print("Index simbólico gerado com sucesso.")