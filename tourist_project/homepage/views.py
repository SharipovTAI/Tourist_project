from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from llama_cpp import Llama

# Загружаем модель один раз при запуске сервера
try:
    llm = Llama(
        model_path="D:/project_workshop/Llama_project/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        n_ctx=2048,
        n_gpu_layers=20,
        verbose=False
    )
    MODEL_LOADED = True
    print("✅ Модель Llama загружена успешно!")
except Exception as e:
    MODEL_LOADED = False
    print(f"❌ Ошибка загрузки модели: {e}")

def index(request):
    return render(request, 'homepage/homepage.html')

@csrf_exempt
def ask_llama(request):
    try:
        data = json.loads(request.body)
        user_question = data.get('question', '')
        
        # Жесткий промпт с четкими инструкциями
        prompt = f"""<|im_start|>system
Ты - туристический помощник. Отвечай ТОЛЬКО на заданный вопрос. 
Не придумывай дополнительные вопросы и диалоги.
Отвечай кратко и по делу.<|im_end|>
<|im_start|>user
{user_question}<|im_end|>
<|im_start|>assistant
"""
        
        response = llm(
            prompt,
            max_tokens=100,      # Короткие ответы
            temperature=0.1,     # Минимум творчества
            top_p=0.7,
            stop=["<|im_end|>", "\n\n", "Пользователь:", "---"],
            echo=False
        )

        answer = response['choices'][0]['text'].strip()
        
        # Обрезаем если модель начала новый вопрос
        if "?" in answer and len(answer.split("?")) > 1:
            answer = answer.split("?")[0] + "?"
        
        return JsonResponse({'answer': answer})
    
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)