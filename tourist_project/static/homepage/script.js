const input = document.getElementById('queryInput');
const sendBtn = document.getElementById('sendBtn');
const chat = document.getElementById('chatArea');

function addMessage(text, isUser = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = isUser ? 'msg user' : 'msg bot';
    
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = text;
    
    messageDiv.appendChild(bubble);
    chat.appendChild(messageDiv);
    chat.scrollTop = chat.scrollHeight;
}

async function sendToLlama(question) {
    try {
        addMessage('Бот думает...', false);
        
        const response = await fetch('/ask-llama/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question: question
            })
        });
        
        const data = await response.json();
        
        chat.removeChild(chat.lastChild);
        
        if (data.answer) {
            addMessage(data.answer, false);
        } else if (data.error) {
            addMessage('Ошибка сервера: ' + data.error, false);
        }
        
    } catch (error) {
        chat.removeChild(chat.lastChild);
        addMessage('Ошибка соединения: ' + error, false);
    }
}

sendBtn.addEventListener('click', function() {
    const question = input.value.trim();
    if (question) {
        addMessage(question, true);
        input.value = '';
        sendToLlama(question);
    }
});

input.addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        sendBtn.click();
    }
});