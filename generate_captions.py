# Имя файла: generate_caption.py

import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from PIL import Image
from transformers import BertTokenizer
import sys
import os

# Проверяем, передан ли путь к изображению
if len(sys.argv) != 2:
    print("Использование: python generate_caption.py path_to_image")
    sys.exit(1)

image_path = sys.argv[1]

# Устройство (CPU или GPU)
os.system("cls")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Токенайзер
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
vocab_size = tokenizer.vocab_size

# Трансформации для изображения
transform = transforms.Compose([
    transforms.ToTensor(),
    # Добавьте нормализацию при необходимости
])

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
        tgt_emb = self.embedding(tgt)
        output = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        output = self.fc(output)
        return output

# Инициализация моделей
embed_dim = 512
num_heads = 8
num_layers = 6

encoder = Encoder(embed_dim).to(device)
decoder = Decoder(embed_dim, num_heads, num_layers, vocab_size).to(device)

# Загрузка обученных моделей
encoder.load_state_dict(torch.load('encoder.pth', map_location=device))
decoder.load_state_dict(torch.load('decoder.pth', map_location=device))

# Функция для создания маски
def create_mask(tgt_seq):
    tgt_len = tgt_seq.size(0)
    mask = torch.triu(torch.ones(tgt_len, tgt_len) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask.to(device)

# Функция для генерации подписи
def generate_caption(encoder, decoder, image, max_length=20):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        image = transform(image).unsqueeze(0).to(device)
        memory = encoder(image)
        generated_ids = [tokenizer.cls_token_id]
        for _ in range(max_length):
            tgt_input = torch.tensor(generated_ids).unsqueeze(1).to(device)
            tgt_mask = create_mask(tgt_input)
            outputs = decoder(tgt_input, memory, tgt_mask=tgt_mask)
            next_token_logits = outputs[-1, 0, :]
            next_token_id = next_token_logits.argmax().item()
            if next_token_id == tokenizer.sep_token_id or next_token_id == tokenizer.pad_token_id:
                break
            generated_ids.append(next_token_id)
        caption = tokenizer.decode(generated_ids[1:], skip_special_tokens=True)
    return caption

# Загрузка изображения
try:
    image = Image.open(image_path).convert('RGB')
except Exception as e:
    print(f"Не удалось открыть изображение: {e}")
    sys.exit(1)

# Генерация подписи
caption = generate_caption(encoder, decoder, image)
print('Сгенерированные теги и описание:', caption)