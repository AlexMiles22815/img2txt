import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import nltk
nltk.download('punkt')

# Загрузка словарей
word2idx = torch.load('word2idx.pth')
idx2word = torch.load('idx2word.pth')
vocab_size = len(word2idx) + 1

# Гиперпараметры (должны совпадать с train.py)
embedding_dim = 256
hidden_dim = 512

# Определение модели (должно совпадать с train.py)
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
        # Получаем признаки изображения и используем их для инициализации LSTM
        features = self.encoder(images)  # (batch_size, hidden_dim)
        embeddings = self.embedding(captions)  # (batch_size, seq_length, embedding_dim)
        
        # Изменяем размер features для использования в качестве h_0 и c_0
        features = features.unsqueeze(0)  # (1, batch_size, hidden_dim)
        hidden = (features, torch.zeros_like(features))  # Инициализируем h_0 и c_0

        # Пропускаем эмбеддинги через LSTM с инициализированным hidden state
        outputs, _ = self.decoder(embeddings, hidden)
        outputs = self.fc(outputs)  # (batch_size, seq_length, vocab_size)
        return outputs

# Загрузка модели
model = ImageCaptioningModel(vocab_size, embedding_dim, hidden_dim)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Предобработка изображения
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Функция для генерации подписи
def generate_caption(image_path, max_length=20):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    features = model.encoder(image)

    caption = []
    input_word = torch.tensor([[word2idx.get('<START>', word2idx['<UNK>'])]])

    hidden = None
    for _ in range(max_length):
        embeddings = model.embedding(input_word)
        outputs, hidden = model.decoder(embeddings, hidden)
        outputs = model.fc(outputs.squeeze(1))
        predicted = outputs.argmax(1)
        predicted_word = idx2word.get(predicted.item(), '<UNK>')

        if predicted_word == '<END>':
            break

        caption.append(predicted_word)
        input_word = predicted.unsqueeze(0)

    return ' '.join(caption)

# Пример использования
image_path = 'images/8368ac2cbd16cfc4be42e71a3f818388.png'
caption = generate_caption(image_path)
print(f"Описание: {caption}")