from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import traceback
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import re

# -----------------------------------------
# Настройки модели LLaMA
# -----------------------------------------
try:
    llm = Llama(
        model_path="D:/project_workshop/Llama_project/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        n_ctx=4096,
        n_gpu_layers=20,
        verbose=False
    )
    MODEL_LOADED = True
    print("✓ Модель LLaMA загружена успешно!")
except Exception as e:
    MODEL_LOADED = False
    print("✗ Ошибка загрузки модели LLaMA:", e)
    print(traceback.format_exc())

# -----------------------------------------
# Загружаем текстовый файл с группировкой
# -----------------------------------------
TEXT_PATH = "D:\\project_workshop\\tourist_project\\homepage\\zapovedniki_optimized_grouped.txt"

# Структура для хранения данных
zapovedniki_data = {}
all_paragraphs = []
paragraph_to_zapovednik = {}

try:
    if os.path.exists(TEXT_PATH):
        with open(TEXT_PATH, 'r', encoding='utf-8') as f:
            content = f.read()
        
        current_zapovednik = None
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('===') and '===' in line:
                match = re.match(r'=== (.*?) ===', line)
                if match:
                    current_zapovednik = match.group(1).strip()
                    zapovedniki_data[current_zapovednik] = []
                continue
            
            if current_zapovednik and line:
                zapovedniki_data[current_zapovednik].append(line)
                all_paragraphs.append(line)
                paragraph_to_zapovednik[len(all_paragraphs)-1] = current_zapovednik
        
        print("="*60)
        print(f"✓ Загружено {len(zapovedniki_data)} заповедников")
        print(f"✓ Всего предложений: {len(all_paragraphs)}")
        
        # Выводим статистику по Таймыру
        if "ТАЙМЫРСКИЙ ЗАПОВЕДНИК" in zapovedniki_data:
            taiymir_count = len(zapovedniki_data["ТАЙМЫРСКИЙ ЗАПОВЕДНИК"])
            geopolit_count = sum(1 for s in zapovedniki_data["ТАЙМЫРСКИЙ ЗАПОВЕДНИК"] 
                               if any(word in s.lower() for word in ['геополит', 'суверенитет', 'ресурс', 'арктик', 'смп', 'северный']))
            print(f"✓ Таймырский заповедник: {taiymir_count} предложений, из них {geopolit_count} про геополитику")
        
    else:
        print(f"✗ Файл не найден: {TEXT_PATH}")
        all_paragraphs = []
        
except Exception as e:
    all_paragraphs = []
    print(f"✗ Ошибка загрузки файла: {e}")

# -----------------------------------------
# Загружаем SentenceTransformer
# -----------------------------------------
try:
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    if all_paragraphs:
        print("Создаем эмбеддинги...")
        doc_embs = embed_model.encode(all_paragraphs, convert_to_numpy=True)
        EMBEDDINGS_LOADED = True
        print(f"✓ Эмбеддинги созданы: {doc_embs.shape}")
    else:
        EMBEDDINGS_LOADED = False
        doc_embs = None
        print("⚠ Нет данных для создания эмбеддингов")
except Exception as e:
    EMBEDDINGS_LOADED = False
    doc_embs = None
    print(f"✗ Ошибка SentenceTransformer: {e}")

# -----------------------------------------
# Словарь для определения заповедников
# -----------------------------------------
ZAPOVEDNIKI_KEYWORDS = {
    'таймырский': 'ТАЙМЫРСКИЙ ЗАПОВЕДНИК',
    'таймыр': 'ТАЙМЫРСКИЙ ЗАПОВЕДНИК',
    # ... остальные заповедники
}

def detect_zapovednik(question):
    """Определяем, о каком заповеднике идет речь"""
    question_lower = question.lower()
    
    for keyword, zapovednik_name in ZAPOVEDNIKI_KEYWORDS.items():
        if keyword in question_lower:
            return zapovednik_name
    
    for zapovednik_name in zapovedniki_data.keys():
        main_word = zapovednik_name.split()[0].lower()
        if main_word in question_lower:
            return zapovednik_name
    
    return None

# -----------------------------------------
# УЛУЧШЕННЫЙ поиск для сложных вопросов
# -----------------------------------------
def get_context_for_question(question, zapovednik_name=None):
    """Умный поиск контекста с учетом сложных запросов"""
    if not EMBEDDINGS_LOADED or doc_embs is None:
        return []
    
    try:
        print(f"\n🔍 ПОИСК: '{question}'")
        
        # Определяем, сложный ли вопрос
        is_complex = any(word in question.lower() for word in ['и ', 'а также', 'также', 'плюс', 'включая', 'кроме'])
        
        # Определяем темы вопроса
        question_lower = question.lower()
        themes = []
        
        if 'геополит' in question_lower or 'геополитический' in question_lower:
            themes.append('геополитика')
        
        # Улучшаем вопрос для поиска
        enhanced_question = question_lower
        
        # Добавляем ключевые слова в зависимости от тем
        if 'геополитика' in themes:
            enhanced_question += " суверенитет ресурсы стратегический арктический северный морской путь безопасность территория граница"
        
        # Определяем, какой заповедник ищем
        if not zapovednik_name:
            zapovednik_name = detect_zapovednik(question)
        
        # Фильтруем предложения по заповеднику
        if zapovednik_name and zapovednik_name in zapovedniki_data:
            # Получаем индексы предложений этого заповедника
            target_indices = []
            for idx, zap in paragraph_to_zapovednik.items():
                if zap == zapovednik_name:
                    target_indices.append(idx)
            
            if not target_indices:
                return []
            
            # Берем эмбеддинги только этого заповедника
            target_embs = doc_embs[target_indices]
            target_paragraphs = [all_paragraphs[i] for i in target_indices]
        else:
            # Ищем по всем предложениям
            target_indices = list(range(len(all_paragraphs)))
            target_embs = doc_embs
            target_paragraphs = all_paragraphs
        
        # Получаем эмбеддинг улучшенного вопроса
        question_emb = embed_model.encode([enhanced_question], convert_to_numpy=True)
        
        # Вычисляем сходство
        similarities = np.dot(target_embs, question_emb.T).flatten()
        
        # Для сложных вопросов берем больше результатов
        top_n = 25 if is_complex else 15
        top_indices_all = np.argsort(similarities)[::-1][:top_n]
        
        # ПЕРВЫЙ ПРОХОД: семантический поиск
        semantic_results = []
        for idx in top_indices_all:
            if similarities[idx] > 0.15:  # Более низкий порог для сложных вопросов
                semantic_results.append(target_paragraphs[idx])
        
        # ВТОРОЙ ПРОХОД: если вопрос сложный и/или содержит темы, ищем по ключевым словам
        keyword_results = []
        if is_complex or themes:
            print(f"  Дополнительный поиск по темам: {themes}")
            
            # Ключевые слова для каждой темы
            theme_keywords = {
                'геополитика': ['геополит', 'суверенитет', 'ресурс', 'стратегическ', 'арктическ', 'северный морской', 'смп', 'безопасност', 'территори', 'границ']
            }
            
            # Собираем все ключевые слова
            all_keywords = []
            for theme in themes:
                if theme in theme_keywords:
                    all_keywords.extend(theme_keywords[theme])
            
            # Если есть ключевые слова, ищем по ним
            if all_keywords:
                for paragraph in target_paragraphs:
                    paragraph_lower = paragraph.lower()
                    # Проверяем, содержит ли параграф хотя бы одно ключевое слово
                    if any(keyword in paragraph_lower for keyword in all_keywords):
                        if paragraph not in semantic_results and paragraph not in keyword_results:
                            keyword_results.append(paragraph)
        
        # Объединяем результаты
        all_results = semantic_results.copy()
        
        # Добавляем ключевые результаты (убираем дубли)
        for result in keyword_results:
            if result not in all_results:
                all_results.append(result)
        
        # Для сложных вопросов берем больше контекста
        max_results = 20 if is_complex else 12
        
        print(f"  Найдено: {len(semantic_results)} семантических + {len(keyword_results)} по ключевым словам = {len(all_results)} всего")
        
        if all_results:
            print("  Примеры контекста:")
            for i, text in enumerate(all_results[:3]):
                print(f"    {i+1}. {text}")
        
        return all_results[:max_results]
        
    except Exception as e:
        print(f"❌ Ошибка поиска: {e}")
        return []

# -----------------------------------------
# Главная страница
# -----------------------------------------
def index(request):
    return render(request, 'homepage/homepage.html')

# -----------------------------------------
# ИСПРАВЛЕННЫЙ API метод
# -----------------------------------------
@csrf_exempt
def ask_llama(request):
    if not MODEL_LOADED:
        return JsonResponse({"error": "Модель LLaMA не загружена"}, status=500)
    
    if not EMBEDDINGS_LOADED:
        return JsonResponse({"error": "Система поиска не загружена"}, status=500)
    
    if not all_paragraphs:
        return JsonResponse({"error": "База знаний пуста"}, status=500)

    if request.method != "POST":
        return JsonResponse({"error": "Метод не разрешен"}, status=405)

    try:
        data = json.loads(request.body)
        user_question = data.get("question", "").strip()

        if not user_question:
            return JsonResponse({"error": "Вопрос не может быть пустым"}, status=400)

        print(f"\n{'='*60}")
        print(f"📨 ВОПРОС: '{user_question}'")
        
        # Определяем заповедник
        detected_zapovednik = detect_zapovednik(user_question)
        
        # Получаем контекст с улучшенным поиском
        context_paragraphs = get_context_for_question(user_question, detected_zapovednik)
        
        if context_paragraphs:
            # Проверяем, есть ли в контексте информация по всем частям вопроса
            question_lower = user_question.lower()
            
            # Проверяем наличие геополитической информации
            if 'геополит' in question_lower:
                geopolit_in_context = any(
                    any(word in p.lower() for word in ['геополит', 'суверенитет', 'ресурс', 'арктическ', 'северный морской', 'смп'])
                    for p in context_paragraphs
                )
                print(f"  Геополитическая информация в контексте: {'ДА' if geopolit_in_context else 'НЕТ'}")
            
            # Формируем контекст с маркировкой
            context_text = "ИНФОРМАЦИЯ О ЗАПОВЕДНИКАХ РОССИИ:\n\n"
            
            # Группируем по заповедникам
            context_by_zapovednik = {}
            for text in context_paragraphs:
                for zap_name, sentences in zapovedniki_data.items():
                    if text in sentences:
                        if zap_name not in context_by_zapovednik:
                            context_by_zapovednik[zap_name] = []
                        context_by_zapovednik[zap_name].append(text)
                        break
            
            # Добавляем информацию по заповедникам
            for zap_name, sentences in context_by_zapovednik.items():
                context_text += f"=== {zap_name} ===\n"
                for sentence in sentences:
                    context_text += f"- {sentence}\n"
                context_text += "\n"
            
            print(f"📚 Контекст: {len(context_paragraphs)} предложений из {len(context_by_zapovednik)} заповедников")
        else:
            context_text = "Информация о заповедниках отсутствует в базе знаний."
            print("⚠ Контекст не найден")
        
        # УСИЛЕННЫЙ промпт для сложных вопросов
        if 'и' in user_question.lower() and user_question.lower().count('и') > 1:
            # Для сложных вопросов с несколькими условиями
            prompt = f"""<|im_start|>system
Ты — эксперт по заповедникам России. Отвечай на вопрос пользователя ПОЛНОСТЬЮ, используя всю предоставленную информацию.

ВАЖНО: Вопрос содержит несколько частей. Ответь на ВСЕ части вопроса.

ИНФОРМАЦИЯ:
{context_text}

ПРАВИЛА ОТВЕТА:
1. Используй ВСЮ предоставленную информацию
2. Ответь на КАЖДУЮ часть вопроса
3. Если информации для какой-то части нет, скажи об этом прямо
4. Не выдумывай факты
5. Структурируй ответ по частям вопроса<|im_end|>
<|im_start|>user
{user_question}<|im_end|>
<|im_start|>assistant
"""
        else:
            # Для простых вопросов
            prompt = f"""<|im_start|>system
Ты — эксперт по заповедникам России. Отвечай на вопрос пользователя, используя предоставленную информацию.

ИНФОРМАЦИЯ:
{context_text}

ПРАВИЛА ОТВЕТА:
1. Используй только предоставленную информацию
2. Если информации недостаточно, скажи об этом
3. Не выдумывай факты
4. Будь точным и конкретным<|im_end|>
<|im_start|>user
{user_question}<|im_end|>
<|im_start|>assistant
"""
        
        print(f"📝 Длина промпта: {len(prompt)} символов")
        print("🤖 Генерирую ответ...")
        
        # Параметры генерации
        response = llm(
            prompt,
            max_tokens=600,  # Больше токенов для сложных ответов
            temperature=0.1,
            top_p=0.85,
            repeat_penalty=1.1,
            stop=["<|im_start|>", "<|im_end|>", "<|eot_id|>"],
            echo=False
        )

        answer = response["choices"][0]["text"].strip()
        
        # Чистка
        for token in ["<|im_start|>", "<|im_end|>", "<|eot_id|>"]:
            answer = answer.replace(token, "").strip()
        
        # Проверяем, ответил ли на все части вопроса
        question_lower = user_question.lower()
        answer_lower = answer.lower()
        
        # Для вопросов с "и" проверяем, что ответ содержит информацию по всем частям
        if 'и' in question_lower and 'геополит' in question_lower:
            # Проверяем, есть ли в ответе геополитическая информация
            geopolit_keywords = ['геополит', 'суверенитет', 'ресурс', 'арктическ', 'северный морской', 'смп']
            has_geopolit = any(keyword in answer_lower for keyword in geopolit_keywords)
            
            # Проверяем, есть ли общая информация о заповеднике
            has_general = any(word in answer_lower for word in ['заповедник', 'таймыр', 'основан', 'площад'])
            
            if not has_geopolit and has_general:
                # Добавляем недостающую информацию
                geopolit_context = []
                for text in context_paragraphs:
                    if any(keyword in text.lower() for keyword in geopolit_keywords):
                        geopolit_context.append(text)
                
                if geopolit_context:
                    answer += "\n\nГеополитические факторы:\n" + "\n".join([f"• {text}" for text in geopolit_context[:3]])
                else:
                    answer += "\n\nПримечание: В предоставленной информации недостаточно данных о геополитических факторах Таймыра."
        
        # Если ответ слишком короткий
        if len(answer) < 30:
            answer = "На основе предоставленной информации:\n" + "\n".join([f"• {p}" for p in context_paragraphs[:5]])
        
        print(f"📤 ОТВЕТ ({len(answer)} символов):")
        print(f"{answer[:300]}..." if len(answer) > 300 else answer)
        print(f"{'='*60}")
        
        return JsonResponse({
            "answer": answer,
            "question": user_question,
            "zapovednik": detected_zapovednik if detected_zapovednik else "Общий вопрос",
            "context_count": len(context_paragraphs),
            "complex_question": 'и' in user_question.lower() and user_question.lower().count('и') > 0
        })

    except Exception as e:
        print(f"❌ Ошибка: {traceback.format_exc()}")
        return JsonResponse({"error": str(e)}, status=500)

# -----------------------------------------
# ДЕБАГ-метод для проверки поиска
# -----------------------------------------
@csrf_exempt
def debug_search(request):
    """Метод для отладки поиска"""
    if request.method != "POST":
        return JsonResponse({"error": "Метод не разрешен"}, status=405)
    
    try:
        data = json.loads(request.body)
        user_question = data.get("question", "").strip()
        
        if not user_question:
            return JsonResponse({"error": "Вопрос не может быть пустым"}, status=400)
        
        # Определяем заповедник
        detected_zapovednik = detect_zapovednik(user_question)
        
        # Получаем контекст
        context_paragraphs = get_context_for_question(user_question, detected_zapovednik)
        
        # Анализируем контекст
        analysis = {
            "question": user_question,
            "detected_zapovednik": detected_zapovednik,
            "total_context_count": len(context_paragraphs),
            "context_by_zapovednik": {},
            "has_geopolit": False,
            "has_general": False
        }
        
        # Группируем по заповедникам
        for text in context_paragraphs:
            for zap_name, sentences in zapovedniki_data.items():
                if text in sentences:
                    if zap_name not in analysis["context_by_zapovednik"]:
                        analysis["context_by_zapovednik"][zap_name] = []
                    analysis["context_by_zapovednik"][zap_name].append(text)
                    break
        
        # Проверяем наличие геополитической информации
        geopolit_keywords = ['геополит', 'суверенитет', 'ресурс', 'арктическ', 'северный морской', 'смп']
        for text in context_paragraphs:
            if any(keyword in text.lower() for keyword in geopolit_keywords):
                analysis["has_geopolit"] = True
                break
        
        # Проверяем наличие общей информации
        general_keywords = ['заповедник', 'основан', 'площад', 'расположен']
        for text in context_paragraphs:
            if any(keyword in text.lower() for keyword in general_keywords):
                analysis["has_general"] = True
                break
        
        # Возвращаем детальный анализ
        return JsonResponse({
            "analysis": analysis,
            "context_preview": context_paragraphs[:10],
            "all_context": context_paragraphs
        })
        
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
