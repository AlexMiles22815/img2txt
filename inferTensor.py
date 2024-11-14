  import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from PIL import Image

def preprocess_image(image_path):
    IMAGE_SIZE = (224, 224)
    img = Image.open(image_path).convert('RGB')
    img = img.resize(IMAGE_SIZE)
    img = np.array(img)
    img = img / 255.0  # Нормализация
    img = np.expand_dims(img, axis=0)  # Добавляем размерность батча
    return img

def generate_description(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat, '')
        if word is None or word == '':
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    final_desc = in_text.replace('startseq', '').replace('endseq', '').strip()
    return final_desc

def main():
    # Загрузка модели
    model = load_model('trained_model.h5')
    print("Модель загружена.")

    # Загрузка токенайзера
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    print("Токенайзер загружен.")

    # Загрузка параметров модели
    with open('model_params.pkl', 'rb') as f:
        params = pickle.load(f)
    max_length = params['max_length']
    print("Параметры модели загружены.")

    # Путь к новому изображению
    image_path = 'path_to_new_image.jpg'

    # Предобработка изображения
    photo = preprocess_image(image_path)

    # Генерация описания
    description = generate_description(model, tokenizer, photo, max_length)
    print("Сгенерированное описание:", description)

if __name__ == '__main__':
    main()
