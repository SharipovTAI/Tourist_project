from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

import json
import os
import numpy as np
import faiss

from llama_cpp import Llama
from sentence_transformers import SentenceTransformer


# ==================== GLOBAL ====================

_rag_assistant = None


# ==================== ASSISTANT ====================

class ReserveRAGAssistant:
    """
    RAG-ассистент по заповедникам.
    LLM используется ТОЛЬКО как генератор текста.
    """

    def __init__(self, model_path, db_path):
        # ---------- Load database ----------
        with open(db_path, "r", encoding="utf-8") as f:
            self.reserves = json.load(f)["reserves"]

        # ---------- Embedding model ----------
        self.embedder = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        # ---------- Build vector index ----------
        self.texts = []
        self.meta = []

        for r in self.reserves:
            text = self._reserve_to_text(r)
            self.texts.append(text)
            self.meta.append(r)

        embeddings = self.embedder.encode(self.texts, show_progress_bar=False)
        embeddings = np.array(embeddings).astype("float32")

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

        # ---------- LLM ----------
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=32,
            n_ctx=2048,
            n_batch=1024,
            temperature=0.3,
            top_p=0.85,
            repeat_penalty=1.1,
            verbose=False,
        )

    # ==================== CORE ====================

    def ask(self, question: str) -> str:
        intent = self._detect_intent(question)

        if intent == "greeting":
            return self._greeting()

        if intent == "recommendation":
            reserves = self._semantic_search(question, top_k=3)
            return self._generate_recommendation(question, reserves)

        reserves = self._semantic_search(question, top_k=1)
        if not reserves:
            return "К сожалению, у меня нет информации для ответа на этот вопрос."

        return self._generate_answer(question, reserves[0])

    # ==================== INTENTS ====================

    def _detect_intent(self, q: str) -> str:
        q = q.lower()

        if any(x in q for x in ["привет", "здравств", "hello", "hi"]):
            return "greeting"

        if any(x in q for x in ["какой выбрать", "посоветуй", "где лучше", "что выбрать"]):
            return "recommendation"

        return "info"

    # ==================== SEARCH ====================

    def _semantic_search(self, query: str, top_k=3):
        q_emb = self.embedder.encode([query]).astype("float32")
        faiss.normalize_L2(q_emb)

        scores, idxs = self.index.search(q_emb, top_k)
        results = []

        for i in idxs[0]:
            if i != -1:
                results.append(self.meta[i])

        return results

    # ==================== GENERATION ====================

    def _generate_answer(self, question, reserve):
        context = self._format_reserve(reserve)

        prompt = f"""
Ты — официальный справочный ассистент по заповедникам России.

СТРОГИЕ ПРАВИЛА:
- Используй ТОЛЬКО информацию ниже
- НЕ добавляй знаний от себя
- Если ответа нет — скажи "Нет данных"

ИНФОРМАЦИЯ:
{context}

ВОПРОС:
{question}

ОТВЕТ:
"""

        out = self.llm(prompt, max_tokens=300, stop=["ВОПРОС:", "ИНФОРМАЦИЯ:"])
        return out["choices"][0]["text"].strip()

    def _generate_recommendation(self, question, reserves):
        blocks = []
        for r in reserves:
            blocks.append(self._format_reserve(r))

        context = "\n\n---\n\n".join(blocks)

        prompt = f"""
Ты — туристический консультант.

СТРОГИЕ ПРАВИЛА:
- Используй ТОЛЬКО информацию ниже
- Сравни варианты
- НЕ придумывай факты

ДАННЫЕ:
{context}

ВОПРОС:
{question}

РЕКОМЕНДАЦИЯ:
"""

        out = self.llm(prompt, max_tokens=400)
        return out["choices"][0]["text"].strip()

    # ==================== HELPERS ====================

    def _greeting(self):
        return (
            "Здравствуйте! Я помогу подобрать заповедник, "
            "рассказать о правилах посещения и интересных местах."
        )

    def _reserve_to_text(self, r):
        parts = [
            r.get("name", ""),
            r.get("description", ""),
            r.get("location", ""),
            r.get("flora_fauna", ""),
            " ".join(r.get("attractions", [])),
        ]
        return " ".join(parts)

    def _format_reserve(self, r):
        lines = [
            f"Название: {r['name']}",
            f"Местоположение: {r.get('location', 'Нет данных')}",
            f"Описание: {r.get('description', 'Нет данных')}",
            f"Флора и фауна: {r.get('flora_fauna', 'Нет данных')}",
        ]

        if r.get("visiting_rules", {}).get("best_time"):
            lines.append(f"Лучшее время посещения: {r['visiting_rules']['best_time']}")

        return "\n".join(lines)


# ==================== INIT ====================

def get_assistant():
    global _rag_assistant

    if _rag_assistant is None:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(BASE_DIR, "model", "mistral-7b-instruct-v0.2.Q4_K_M.gguf")
        db_path = os.path.join(BASE_DIR, "reserves_database.json")

        _rag_assistant = ReserveRAGAssistant(model_path, db_path)

    return _rag_assistant


# ==================== DJANGO VIEWS ====================

@csrf_exempt
def ask_llama(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=400)

    data = json.loads(request.body)
    question = data.get("question", "").strip()

    if not question:
        return JsonResponse({"answer": "Задайте вопрос"})

    assistant = get_assistant()
    answer = assistant.ask(question)

    return JsonResponse({"answer": answer})


def index(request):
    get_assistant()
    return render(request, "homepage/homepage.html")
