import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import argparse
import nltk
nltk.download('punkt')

# Аргументы командной строки
parser = argparse.ArgumentParser(description='Generate caption for an image using a trained model.')
parser.add_argument('--image_path', type=str, required=True, help='Path to the input image.')
args = parser.parse_args()

# Загрузка словарей
word2idx = torch.load('word2idx.pth')
idx2word = torch.load('idx2word.pth')
vocab_size = len(word2idx)

# Гиперпараметры (должны совпадать с теми, что использовались при обучении)
embedding_dim = 256
hidden_dim = 512
max_seq_length = 20  # Максимальная длина генерируемой подписи

# Определение модели (должно совпадать с тем, что использовалось при обучении)
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
    
    def forward(self, features, captions, hidden):
        embeddings = self.embedding(captions)
        outputs, hidden = self.decoder(embeddings, hidden)
        outputs = self.fc(outputs)
        return outputs, hidden

# Загрузка модели
model = ImageCaptioningModel(vocab_size, embedding_dim, hidden_dim)
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
model.eval()

# Предобработка изображения
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Функция для генерации описания
def generate_caption(model, image_path, word2idx, idx2word, max_length=20):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)

    # Получаем признаки изображения
    features = model.encoder(image)  # (batch_size=1, hidden_dim)
    features = features.unsqueeze(0)  # (num_layers=1, batch_size=1, hidden_dim)
    hidden = (features, torch.zeros_like(features))  # Инициализируем hidden state

    # Начинаем генерацию с токена <START>
    input_word = torch.tensor([[word2idx['<START>']]])

    generated_caption = []

    for _ in range(max_length):
        outputs, hidden = model.forward(None, input_word, hidden)
        outputs = outputs.squeeze(1)  # (batch_size=1, vocab_size)
        predicted_idx = outputs.argmax(dim=1).item()
        predicted_word = idx2word.get(predicted_idx, '<UNK>')

        if predicted_word == '<END>':
            break

        generated_caption.append(predicted_word)
        input_word = torch.tensor([[predicted_idx]])

    return ' '.join(generated_caption)

# Генерация описания для заданного изображения
caption = generate_caption(model, args.image_path, word2idx, idx2word)
print(f"Описание: {caption}")
torch.cuda.empty_cache()