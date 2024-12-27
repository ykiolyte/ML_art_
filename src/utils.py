# src/utils.py

import os
import torch

def save_checkpoint(state, checkpoint_dir='checkpoints'):
    filepath = os.path.join(checkpoint_dir, 'best_model.pth')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    torch.save(state, filepath)
    print(f"Модель сохранена в {filepath}")

def load_checkpoint(model, checkpoint_dir='checkpoints'):
    filepath = os.path.join(checkpoint_dir, 'best_model.pth')
    if os.path.exists(filepath):
        print(f"Загружается модель из {filepath}")
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        print("Модель успешно загружена.")
    else:
        print(f"Файл контрольной точки не найден в {filepath}.")

def get_pos_weight(dataset):
    # Если вам нужен pos_weight для функции потерь
    labels = dataset.labels
    pos_count = sum(labels)
    neg_count = len(labels) - pos_count
    pos_weight = torch.tensor(neg_count / pos_count, dtype=torch.float32)
    return pos_weight
