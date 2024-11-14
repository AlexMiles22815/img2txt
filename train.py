import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_preprocessing import ImageCaptionDataset
import torchvision.transforms as transforms
from collections import Counter
import nltk
nltk.download('punkt')

# Выбор устройства (CPU или GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используемое устройство: {device}")

# Гиперпараметры
batch_size = 32
num_epochs = 500
learning_rate = 0.001
embedding_dim = 256
hidden_dim = 512

# Предобработка изображений
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Загрузка датасета
dataset = ImageCaptionDataset('images', 'captions', transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Создание словаря
def build_vocab(captions, threshold):
    counter = Counter()
    for caption in captions:
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)
    
    words = [word for word, cnt in counter.items() if cnt >= threshold]
    word2idx = {word: idx+1 for idx, word in enumerate(words)}  # +1 для зарезервирования idx=0 для <PAD>
    word2idx['<PAD>'] = 0
    word2idx['<UNK>'] = len(word2idx)
    word2idx['<START>'] = len(word2idx)
    word2idx['<END>'] = len(word2idx)
    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word

# Собираем все подписи
all_captions = [caption for _, caption in dataset]
word2idx, idx2word = build_vocab(all_captions, threshold=1)
vocab_size = len(word2idx)

# Определяем максимальную длину последовательности
def get_max_seq_length(captions):
    max_length = 0
    for caption in captions:
        tokens = ['<START>'] + nltk.tokenize.word_tokenize(caption.lower()) + ['<END>']
        if len(tokens) > max_length:
            max_length = len(tokens)
    return max_length

max_seq_length = get_max_seq_length(all_captions)

# Токенизация и паддинг
def caption_to_tensor(caption, word2idx, max_seq_length):
    tokens = ['<START>'] + nltk.tokenize.word_tokenize(caption.lower()) + ['<END>']
    tensor = [word2idx.get(token, word2idx['<UNK>']) for token in tokens]
    # Паддинг
    tensor += [word2idx['<PAD>']] * (max_seq_length - len(tensor))
    return torch.tensor(tensor, dtype=torch.long)

# Определение модели
class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 112 * 112, hidden_dim),
            nn.ReLU()
        )
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.decoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, images, captions):
        features = self.encoder(images)  # (batch_size, hidden_dim)
        embeddings = self.embedding(captions)  # (batch_size, seq_length, embedding_dim)
        
        # Инициализируем скрытое состояние LSTM признаками изображения
        features = features.unsqueeze(0)  # (1, batch_size, hidden_dim)
        hidden = (features, torch.zeros_like(features))  # (h_0, c_0)
        
        outputs, _ = self.decoder(embeddings, hidden)
        outputs = self.fc(outputs)  # (batch_size, seq_length, vocab_size)
        return outputs

# Создание модели
model = ImageCaptioningModel(vocab_size, embedding_dim, hidden_dim).to(device)

# Определение функции потерь и оптимизатора
criterion = nn.CrossEntropyLoss(ignore_index=word2idx['<PAD>'])
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Цикл тренировки
for epoch in range(num_epochs):
    for images, captions in dataloader:
        # Перемещаем данные на устройство
        images = images.to(device)
        captions_input = [caption_to_tensor(caption, word2idx, max_seq_length) for caption in captions]
        captions_batch = torch.stack(captions_input).to(device)  # (batch_size, max_seq_length)
        
        # Прямой проход
        outputs = model(images, captions_batch[:, :-1])
        
        # Вычисление потерь
        loss = criterion(outputs.reshape(-1, vocab_size), captions_batch[:, 1:].reshape(-1))
        
        # Обратный проход и оптимизация
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Эпоха [{epoch+1}/{num_epochs}], Потеря: {loss.item():.4f}")

# Сохранение модели и словаря
torch.save(model.state_dict(), 'model.pth')
torch.save(word2idx, 'word2idx.pth')
torch.save(idx2word, 'idx2word.pth')