
# Имя файла: train.py

import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import BertTokenizer

# Путь к папкам
IMAGE_DIR = 'images'    
CAPTION_DIR = 'captions'

# Устройство (CPU или GPU)
# os.system("cls")
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cuda"
print("Using device:", device)


# Токенайзер
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
vocab_size = tokenizer.vocab_size

# Кастомный датасет
class ImageCaptionDataset(Dataset):
    def __init__(self, image_dir, caption_dir, transform=None):
        self.image_dir = image_dir
        self.caption_dir = caption_dir
        self.transform = transform

        # Получаем словари файлов без расширений
        image_files = {os.path.splitext(f)[0]: f for f in os.listdir(image_dir)}
        caption_files = {os.path.splitext(f)[0]: f for f in os.listdir(caption_dir)}

        # Оставляем только те файлы, которые есть и в изображениях, и в подписях
        common_files = set(image_files.keys()).intersection(set(caption_files.keys()))
        self.samples = []

        for basename in common_files:
            image_filename = image_files[basename]
            caption_filename = caption_files[basename]
            image_path = os.path.join(image_dir, image_filename)
            caption_path = os.path.join(caption_dir, caption_filename)
            if os.path.isfile(image_path) and os.path.isfile(caption_path):
                self.samples.append((image_path, caption_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, caption_path = self.samples[idx]
        # Открываем и обрабатываем изображение
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        # Читаем подпись
        with open(caption_path, 'r', encoding='utf-8') as f:
            caption = f.read().strip()
        # Токенизируем подпись
        tokens = tokenizer.encode(caption, add_special_tokens=True)
        tokens = torch.tensor(tokens)
        return image, tokens

# Остальная часть кода остается без изменений

# Функция collate_fn для DataLoader
def collate_fn(batch):
    images, captions = zip(*batch)
    # Объединяем изображения в батч
    images = torch.stack(images)
    # Выравниваем подписи по длине
    captions = torch.nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=tokenizer.pad_token_id)
    return images, captions

# Трансформации для изображений
transform = transforms.Compose([
    transforms.ToTensor(),
    # Добавьте нормализацию при необходимости
])

# Создаем датасет и загрузчик данных
dataset = ImageCaptionDataset(IMAGE_DIR, CAPTION_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# Определение моделей
class Encoder(nn.Module):
    def __init__(self, embed_dim):
        super(Encoder, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        # Добавляем сверточный слой для приведения размерности к embed_dim
        self.conv = nn.Conv2d(2048, embed_dim, kernel_size=1)
    def forward(self, images):
        features = self.resnet(images)  # [batch_size, 2048, h, w]
        features = self.conv(features)   # [batch_size, embed_dim, h, w]
        features = features.flatten(2).permute(2, 0, 1)  # [seq_len, batch_size, embed_dim]
        return features

class Decoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, vocab_size):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, vocab_size)
    def forward(self, tgt, memory, tgt_mask=None):
        tgt_emb = self.embedding(tgt)  # [seq_len, batch_size, embed_dim]
        output = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        output = self.fc(output)
        return output

# Инициализация моделей
embed_dim = 512
num_heads = 8   
num_layers = 6

encoder = Encoder(embed_dim).to(device)
decoder = Decoder(embed_dim, num_heads, num_layers, vocab_size).to(device)

# Оптимизатор и функция потерь
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(params, lr=0.0001)

# Функция для создания маски
def create_mask(tgt_seq):
    tgt_len = tgt_seq.size(0)
    mask = torch.triu(torch.ones(tgt_len, tgt_len) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask.to(device)

# Цикл обучения
num_epochs = 150

for epoch in range(num_epochs):
    encoder.train()
    decoder.train()
    total_loss = 0
    for images, captions in dataloader:
        images = images.to(device)
        captions = captions.to(device)
        optimizer.zero_grad()
        # Пропускаем изображения через кодировщик
        memory = encoder(images)  # [seq_len, batch_size, features]
        # Подготавливаем входы и цели для декодера
        tgt_input = captions[:, :-1].transpose(0, 1)  # [seq_len, batch_size]
        tgt_output = captions[:, 1:].transpose(0, 1)  # [seq_len, batch_size]
        tgt_mask = create_mask(tgt_input)
        # Пропускаем через декодер
        outputs = decoder(tgt_input, memory, tgt_mask=tgt_mask)
        # Вычисляем функцию потерь
        loss = criterion(outputs.reshape(-1, vocab_size), tgt_output.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

# Сохранение моделей
torch.save(encoder.state_dict(), 'encoder.pth')
torch.save(decoder.state_dict(), 'decoder.pth')
