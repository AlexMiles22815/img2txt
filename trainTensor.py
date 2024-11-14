import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import string
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import mixed_precision
import pickle

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Включение смешанной точности
mixed_precision.set_global_policy('mixed_float16')

# Параметры предобработки
MAX_VOCAB_SIZE = 5000  # Максимальный размер словаря
MAX_CAPTION_LENGTH = 50  # Максимальная длина описания
IMAGE_SIZE = (224, 224)  # Размер изображений для модели

def configure_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Установить динамическое выделение памяти
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Используется GPU: {gpus}")
        except RuntimeError as e:
            print(e)
    else:
        print("GPU не обнаружен. Обучение будет происходить на CPU.")

def load_data(image_dir, caption_dir):
    # Создаем словарь для хранения путей к изображениям и их описаниям
    data = {}
    
    # Получаем список всех файлов в директориях
    image_files = os.listdir(image_dir)
    caption_files = os.listdir(caption_dir)
    
    # Создаем словарь: базовое имя файла -> путь к изображению
    image_dict = {}
    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        image_path = os.path.join(image_dir, img_file)
        image_dict[base_name] = image_path
    
    # Сопоставляем изображения и описания
    for cap_file in caption_files:
        base_name = os.path.splitext(cap_file)[0]
        if base_name in image_dict:
            caption_path = os.path.join(caption_dir, cap_file)
            data[base_name] = {'image': image_dict[base_name], 'caption': caption_path}
        else:
            print(f"Предупреждение: для {cap_file} не найдено соответствующее изображение.")
    
    return data

def preprocess_captions(captions):
    # Объединяем все описания для создания словаря
    all_captions = []
    table = str.maketrans('', '', string.punctuation)
    
    for caption in captions:
        caption = caption.lower()
        caption = caption.translate(table)
        caption = 'startseq ' + caption + ' endseq'
        all_captions.append(caption)
    
    # Инициализируем токенайзер и обучаем его
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token='unk')
    tokenizer.fit_on_texts(all_captions)
    
    return all_captions, tokenizer

def preprocess_images(image_paths):
    images = []
    for img_path in tqdm(image_paths, desc="Предобработка изображений"):
        # Открываем и преобразуем изображение
        img = Image.open(img_path).convert('RGB')
        img = img.resize(IMAGE_SIZE)
        img = np.array(img)
        img = img / 255.0  # Нормализация
        images.append(img)
    return np.array(images)

def create_sequences(tokenizer, max_length, images, captions):
    X1, X2, y = [], [], []
    vocab_size = len(tokenizer.word_index) + 1
    for i, caption in enumerate(captions):
        seq = tokenizer.texts_to_sequences([caption])[0]
        img = images[i]
        for j in range(1, len(seq)):
            # Входная последовательность слов
            in_seq = seq[:j]
            # Следующее слово, которое нужно предсказать
            out_seq = seq[j]
            # Паддинг входной последовательности
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            # Добавление в обучающие данные
            X1.append(img)
            X2.append(in_seq)
            y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

def main():
    parser = argparse.ArgumentParser(description='Предобработка датасета для img2txt модели.')
    parser.add_argument('--image_dir', type=str, required=True, help='Директория с изображениями.')
    parser.add_argument('--caption_dir', type=str, required=True, help='Директория с описаниями.')
    args = parser.parse_args()
    
    # Настройка GPU
    configure_gpu()
    
    # Загрузка данных
    data = load_data(args.image_dir, args.caption_dir)
    print(f"Найдено {len(data)} пар изображение-описание.")
    
    # Предобработка изображений и описаний
    image_paths = [item['image'] for item in data.values()]
    caption_paths = [item['caption'] for item in data.values()]
    
    X_images = preprocess_images(image_paths)
    
    captions = []
    for cap_path in caption_paths:
        with open(cap_path, 'r', encoding='utf-8') as file:
            caption = file.read().strip()
            captions.append(caption)
    
    captions, tokenizer = preprocess_captions(captions)
    
    # Создание последовательностей
    max_length = MAX_CAPTION_LENGTH - 1
    X1, X2, y = create_sequences(tokenizer, max_length, X_images, captions)
    
    # Создание и обучение модели
    vocab_size = min(MAX_VOCAB_SIZE, len(tokenizer.word_index) + 1)
    model = build_model(vocab_size, max_length)
    
    # Компиляция модели с использованием оптимизатора, поддерживающего смешанную точность
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    optimizer = mixed_precision.LossScaleOptimizer(optimizer)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer)
    
    # Обучение модели
    model.fit([X1, X2], y, epochs=10, batch_size=64)

    # Сохранение модели
    model.save('trained_model.h5')
    print("Модель сохранена в файл 'trained_model.h5'")

    # Сохранение токенайзера
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    print("Токенайзер сохранен в файл 'tokenizer.pkl'")

    # Сохранение параметров модели
    model_params = {
        'vocab_size': vocab_size,
        'max_length': max_length
    }
    with open('model_params.pkl', 'wb') as f:
        pickle.dump(model_params, f)
    print("Параметры модели сохранены в файл 'model_params.pkl'")

def build_model(vocab_size, max_length):
    # Экстрактор признаков изображения
    inputs_image = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x1 = tf.keras.applications.InceptionV3(include_top=False, pooling='avg')(inputs_image)
    x1 = tf.keras.layers.Dense(256, activation='relu')(x1)
    
    # Генератор текста
    inputs_text = tf.keras.Input(shape=(max_length,))
    x2 = tf.keras.layers.Embedding(vocab_size, 256, mask_zero=True)(inputs_text)
    x2 = tf.keras.layers.LSTM(256)(x2)
    
    # Объединение слоев
    x = tf.keras.layers.concatenate([x1, x2])
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    outputs = tf.keras.layers.Dense(vocab_size, activation='softmax', dtype='float32')(x)
    
    model = tf.keras.Model(inputs=[inputs_image, inputs_text], outputs=outputs)
    return model

if __name__ == '__main__':
    main()
