# src/data_preparation.py

import os
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class PS2Dataset(Dataset):
    def __init__(self, data_dir, transform=None, phase='train'):
        self.data_dir = data_dir
        self.transform = transform
        self.phase = phase
        self.images = []
        self.labels = []

        # Определение классов
        self.classes = ['normal', 'defective']  # 0: normal, 1: defective

        # Загрузка данных и меток
        for label in self.classes:
            class_dir = os.path.join(data_dir, label)
            if not os.path.isdir(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if os.path.isfile(img_path) and img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(img_path)
                    self.labels.append(1 if label == 'defective' else 0)

        # Обработка дисбаланса классов путем oversampling
        if self.phase == 'train':
            self.balance_dataset()

    def balance_dataset(self):
        from collections import Counter
        from sklearn.utils import resample

        # Разделяем данные по классам
        data = list(zip(self.images, self.labels))
        data_normal = [d for d in data if d[1] == 0]
        data_defective = [d for d in data if d[1] == 1]

        # Находим класс с наибольшим количеством образцов
        max_count = max(len(data_normal), len(data_defective))

        # Oversampling меньшего класса
        if len(data_normal) < max_count:
            data_normal = resample(data_normal, replace=True, n_samples=max_count, random_state=42)
        elif len(data_defective) < max_count:
            data_defective = resample(data_defective, replace=True, n_samples=max_count, random_state=42)

        # Объединяем данные и перемешиваем
        balanced_data = data_normal + data_defective
        random.shuffle(balanced_data)

        self.images, self.labels = zip(*balanced_data)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.images[idx]).convert('RGB'))
        label = self.labels[idx]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        label = torch.tensor(label, dtype=torch.float32)

        return image, label

def get_transforms(phase):
    if phase == 'train':
        return A.Compose([
            A.Resize(512, 512),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0)),
                A.MotionBlur(blur_limit=5),
                A.MedianBlur(blur_limit=5),
            ], p=0.5),
            A.OneOf([
                A.ElasticTransform(),
                A.GridDistortion(),
                A.OpticalDistortion(),
            ], p=0.5),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:  # val or test
        return A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
