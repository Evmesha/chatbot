import json
import random
import re
import nltk
import telegram
import nest_asyncio
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split  # Импортируем для разделения данных
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters

# Загружаем данные из файла
with open('intents.json', 'r') as json_file:
    data = json.load(json_file)

X = []
y = []

for name in data:
    for phrase in data[name]['examples']:
        X.append(phrase)
        y.append(name)
    for phrase in data[name]['responses']:
        X.append(phrase)
        y.append(name)

def clean_up(text):
    # Преобразуем текст к нижнему регистру
    text = text.lower()
    # Удаляем все, что не является буквой или пробелом
    re_not_word = r'[^\w\s]'
    text = re.sub(re_not_word, '', text)
    return text

# Очистка данных
X = [clean_up(text) for text in X]

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Векторизация текста
vectorizer = CountVectorizer()
vectorizer.fit(X_train)  # Обучаем векторизатор на обучающей выборке
X_train_vec = vectorizer.transform(X_train)
X_test_vec = vectorizer.transform(X_test)  # Преобразуем тестовую выборку

# Создаем и обучаем модель
model_mlp = MLPClassifier()
model_mlp.fit(X_train_vec, y_train)

# Оценка модели на тестовой выборке
score = model_mlp.score(X_test_vec, y_test)
print("Accuracy on test set:", score)
def get_intent(text):
    # сначала преобразуем текст в числа

    text_vec = vectorizer.transform([text])
    # берем элемент номер 0 - для того, чтобы избавиться от формата "список", который необходим для векторизации и машинного обучения
    return model_mlp.predict(text_vec)[0]
def get_response(intent):
    return random.choice(data[intent]['responses'])
def bot(text):
    text = clean_up(text)
    intent = get_intent(text)
    answer = get_response(intent)
    return answer
text = ""

nest_asyncio.apply()
TOKEN = ''
async def reply(update: Update, context) -> None:

    user_text = update.message.text
    reply = bot(user_text)
    print('<', user_text)
    print('>', reply)

    await update.message.reply_text(reply)  # Ответ пользователю в чат ТГ

# Создаем объект приложения - связываем его с токеном
app = ApplicationBuilder().token(TOKEN).build()

# Создаем обработчик текстовых сообщений
handler = MessageHandler(filters.Text(), reply)

# Добавляем обработчик в приложение
app.add_handler(handler)

# Запускаем приложение: бот крутится, пока крутится колесико выполнения слева ячейки)
app.run_polling()

