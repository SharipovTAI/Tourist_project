from llama_cpp import Llama
import llama_cpp

# Сначала проверим версию библиотеки
print("llama-cpp-python version:", llama_cpp.__version__)

# Затем создаем модель
llm = Llama(
    model_path="models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    n_ctx=2048,
    n_gpu_layers=20,  # максимум GPU
    verbose=False
)


# Тест запроса
response = llm(
    "Привет, что такое геополитические факторы?",
    max_tokens=200,
    echo=False  # не повторять промпт в ответе
)

print("\nОтвет модели:")
print(response["choices"][0]["text"])