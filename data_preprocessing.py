import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ImageCaptionDataset(Dataset):
    def __init__(self, images_dir, captions_dir, transform=None):
        self.images_dir = images_dir
        self.captions_dir = captions_dir
        self.transform = transform
        self.image_caption_pairs = []

        # Поддерживаемые расширения изображений
        image_extensions = ('.jpg', '.jpeg', '.png')

        # Получаем списки файлов изображений и описаний
        image_files = {}
        for f in os.listdir(images_dir):
            if f.lower().endswith(image_extensions):
                base_name = os.path.splitext(f)[0]
                image_files[base_name] = f

        caption_files = {}
        for f in os.listdir(captions_dir):
            base_name = os.path.splitext(f)[0]
            caption_files[base_name] = f

        # Ищем общие базовые имена файлов
        common_files = set(image_files.keys()).intersection(caption_files.keys())

        if not common_files:
            print("Не найдено общих файлов между директориями 'images' и 'captions'.")

        for base_name in common_files:
            image_path = os.path.join(images_dir, image_files[base_name])
            caption_path = os.path.join(captions_dir, caption_files[base_name])

            if os.path.isfile(image_path) and os.path.isfile(caption_path):
                self.image_caption_pairs.append((image_path, caption_path))
            else:
                print(f"Пропускаем файл {base_name} из-за отсутствия изображения или описания.")

        print(f"Найдено {len(self.image_caption_pairs)} пар изображений и описаний.")

    def __len__(self):
        return len(self.image_caption_pairs)

    def __getitem__(self, idx):
        image_path, caption_path = self.image_caption_pairs[idx]

        # Загружаем и преобразуем изображение
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Читаем описание
        with open(caption_path, 'r', encoding='utf-8') as f:
            caption = f.read()

        return image, caption