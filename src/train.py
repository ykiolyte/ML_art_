# src/train.py

import os
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data_preparation import PS2Dataset, get_transforms
from src.model import PS2Net, StackingMetaModel
from src.utils import save_checkpoint
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from tqdm import tqdm
from torch.amp import GradScaler, autocast

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

def train_model():
    # Параметры
    num_epochs = 25
    batch_size = 8 
    learning_rate = 1e-4
    patience = 5
    checkpoint_dir = 'checkpoints'

    # Устройство
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")
    torch.backends.cudnn.benchmark = True  # Для ускорения

    # Датасеты и загрузчики данных
    print("Загружаем датасеты...")
    train_dataset = PS2Dataset(data_dir='data/train', transform=get_transforms('train'), phase='train')
    val_dataset = PS2Dataset(data_dir='data/val', transform=get_transforms('val'), phase='val')
    print(f"Количество обучающих примеров: {len(train_dataset)}")
    print(f"Количество валидационных примеров: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8,
                              pin_memory=True, prefetch_factor=2, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8,
                            pin_memory=True, prefetch_factor=2, persistent_workers=True)
    print("Датасеты загружены.")

    # Список моделей для ансамблирования
    model_names = ['efficientnet_v2_s', 'resnet50', 'convnext_tiny']
    models = []
    optimizers = []
    schedulers = []
    criterions = []
    scalers = []  # Отдельный GradScaler для каждой модели

    for model_name in model_names:
        print(f"Инициализируем модель {model_name}...")
        model = PS2Net(model_name=model_name).to(device)
        criterion = FocalLoss()

        # Понижаем скорость обучения для convnext_tiny
        if model_name == 'convnext_tiny':
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate * 0.1, weight_decay=1e-4)
        else:
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        scaler = GradScaler()  # Отдельный GradScaler для каждой модели

        models.append(model)
        optimizers.append(optimizer)
        schedulers.append(scheduler)
        criterions.append(criterion)
        scalers.append(scaler)

    # Логирование
    writer = SummaryWriter(log_dir='logs')
    print("Логирование настроено.")

    # Для сохранения лучших моделей
    best_auc = [0.0] * len(models)
    patience_counters = [0] * len(models)

    # Тренировка базовых моделей
    for epoch in range(num_epochs):
        print(f"Начало эпохи {epoch+1}/{num_epochs}")
        for model in models:
            model.train()

        running_loss = [0.0] * len(models)
        all_labels = []
        all_preds = [[] for _ in range(len(models))]
        all_binary_preds = [[] for _ in range(len(models))]

        for images, labels in tqdm(train_loader, desc=f"Эпоха {epoch+1}"):
            images = images.to(device, non_blocking=True)
            labels = labels.unsqueeze(1).float().to(device, non_blocking=True)

            for i, model in enumerate(models):
                optimizers[i].zero_grad()
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(images)
                    loss = criterions[i](outputs, labels)
                    if torch.isnan(loss):
                        print(f"Предупреждение: Получена NaN потеря на модели {model_names[i]} в эпохе {epoch+1}")
                        continue  # Пропускаем обновление для этой модели
                scalers[i].scale(loss).backward()
                scalers[i].unscale_(optimizers[i])  # Размасштабируем градиенты для проверки

                # Обрезка градиентов
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

                scalers[i].step(optimizers[i])
                scalers[i].update()

                running_loss[i] += loss.item() * images.size(0)

                probs = torch.sigmoid(outputs).detach().cpu().numpy()
                if np.isnan(probs).any():
                    print(f"Предупреждение: Получены NaN предсказания на модели {model_names[i]} в эпохе {epoch+1}")
                    continue  # Пропускаем обработку этой модели

                all_preds[i].extend(probs.flatten())
                binary_preds = (probs >= 0.5).astype(int)
                all_binary_preds[i].extend(binary_preds.flatten())

            all_labels.extend(labels.cpu().numpy().flatten())

        # Вычисление метрик для каждой модели
        for i, model_name in enumerate(model_names):
            epoch_loss = running_loss[i] / len(train_dataset)
            preds = np.array(all_preds[i])
            binary_preds = np.array(all_binary_preds[i])
            labels_np = np.array(all_labels)

            # Проверка на NaN в предсказаниях
            if np.isnan(preds).any():
                print(f"Предупреждение: NaN значения в предсказаниях модели {model_name} в эпохе {epoch+1}")
                patience_counters[i] += 1
                continue  # Переходим к следующей модели

            train_auc = roc_auc_score(labels_np, preds)
            train_accuracy = accuracy_score(labels_np, binary_preds)
            train_precision = precision_score(labels_np, binary_preds)
            train_recall = recall_score(labels_np, binary_preds)
            train_f1 = f1_score(labels_np, binary_preds)

            writer.add_scalar(f'Loss/train_{model_name}', epoch_loss, epoch)
            writer.add_scalar(f'AUC/train_{model_name}', train_auc, epoch)
            writer.add_scalar(f'Accuracy/train_{model_name}', train_accuracy, epoch)
            writer.add_scalar(f'Precision/train_{model_name}', train_precision, epoch)
            writer.add_scalar(f'Recall/train_{model_name}', train_recall, epoch)
            writer.add_scalar(f'F1_Score/train_{model_name}', train_f1, epoch)

            val_loss, val_auc, val_accuracy, val_precision, val_recall, val_f1 = validate(models[i], val_loader, criterions[i], device)
            writer.add_scalar(f'Loss/val_{model_name}', val_loss, epoch)
            writer.add_scalar(f'AUC/val_{model_name}', val_auc, epoch)
            writer.add_scalar(f'Accuracy/val_{model_name}', val_accuracy, epoch)
            writer.add_scalar(f'Precision/val_{model_name}', val_precision, epoch)
            writer.add_scalar(f'Recall/val_{model_name}', val_recall, epoch)
            writer.add_scalar(f'F1_Score/val_{model_name}', val_f1, epoch)

            print(f'Модель {model_name} - Epoch {epoch+1}/{num_epochs}, '
                  f'Loss: {epoch_loss:.4f}, AUC: {train_auc:.4f}, Accuracy: {train_accuracy:.4f}, '
                  f'Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}, Val Accuracy: {val_accuracy:.4f}, '
                  f'Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}')

            # Шаг планировщика
            schedulers[i].step()

            # Early Stopping и сохранение лучшей модели
            if val_auc > best_auc[i]:
                best_auc[i] = val_auc
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': models[i].state_dict(),
                    'optimizer': optimizers[i].state_dict(),
                }, checkpoint_dir=os.path.join(checkpoint_dir, model_name))
                patience_counters[i] = 0
                print(f"Лучший результат для {model_name} достигнут в эпохе {epoch+1}, модель сохранена.")
            else:
                patience_counters[i] += 1
                if patience_counters[i] >= patience:
                    print(f"Early stopping для модели {model_name}")

        # Проверяем, не остановились ли все модели
        if all(counter >= patience for counter in patience_counters):
            print("Early stopping для всех моделей")
            break

    writer.close()
    print("Обучение базовых моделей завершено.")

    # Обучение мета-модели для стекинга
    print("Обучение мета-модели для стекинга...")
    train_meta_model(models, device, checkpoint_dir)
    print("Обучение мета-модели завершено.")

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_binary_preds = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.unsqueeze(1).float().to(device, non_blocking=True)

            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(probs.flatten())
            binary_preds = (probs >= 0.5).astype(int)
            all_binary_preds.extend(binary_preds.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    epoch_loss = running_loss / len(val_loader.dataset)
    all_preds = np.array(all_preds)
    all_binary_preds = np.array(all_binary_preds)
    all_labels = np.array(all_labels)

    val_auc = roc_auc_score(all_labels, all_preds)
    val_accuracy = accuracy_score(all_labels, all_binary_preds)
    val_precision = precision_score(all_labels, all_binary_preds)
    val_recall = recall_score(all_labels, all_binary_preds)
    val_f1 = f1_score(all_labels, all_binary_preds)

    return epoch_loss, val_auc, val_accuracy, val_precision, val_recall, val_f1

def train_meta_model(models, device, checkpoint_dir):
    # Создаем новый датасет без аугментаций для мета-модели
    batch_size = 16
    train_dataset_no_aug = PS2Dataset(data_dir='data/train', transform=get_transforms('val'), phase='val')
    val_dataset_no_aug = PS2Dataset(data_dir='data/val', transform=get_transforms('val'), phase='val')
    train_loader_no_aug = DataLoader(train_dataset_no_aug, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=2)
    val_loader_no_aug = DataLoader(val_dataset_no_aug, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=2)

    # Собираем предсказания базовых моделей на обучающем наборе
    print("Сбор предсказаний базовых моделей на обучающем наборе...")
    models_preds = [[] for _ in range(len(models))]
    all_labels = []

    for model in models:
        model.eval()

    with torch.no_grad():
        for images, labels in tqdm(train_loader_no_aug, desc="Предсказания на обучающем наборе"):
            images = images.to(device, non_blocking=True)
            labels = labels.unsqueeze(1).float().to(device, non_blocking=True)

            for idx, model in enumerate(models):
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(images)
                    probs = torch.sigmoid(outputs)
                models_preds[idx].append(probs.cpu().numpy())

            all_labels.extend(labels.cpu().numpy().flatten())

    # Объединяем предсказания моделей
    train_meta_features = np.hstack([np.vstack(preds) for preds in models_preds])
    train_meta_labels = np.array(all_labels)

    # То же самое для валидационного набора
    print("Сбор предсказаний базовых моделей на валидационном наборе...")
    val_models_preds = [[] for _ in range(len(models))]
    val_labels = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader_no_aug, desc="Предсказания на валидационном наборе"):
            images = images.to(device, non_blocking=True)
            labels = labels.unsqueeze(1).float().to(device, non_blocking=True)

            for idx, model in enumerate(models):
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(images)
                    probs = torch.sigmoid(outputs)
                val_models_preds[idx].append(probs.cpu().numpy())

            val_labels.extend(labels.cpu().numpy().flatten())

    val_meta_features = np.hstack([np.vstack(preds) for preds in val_models_preds])
    val_meta_labels = np.array(val_labels)

    # Преобразуем в тензоры
    train_meta_features = torch.tensor(train_meta_features, dtype=torch.float32).to(device)
    train_meta_labels = torch.tensor(train_meta_labels, dtype=torch.float32).unsqueeze(1).to(device)
    val_meta_features = torch.tensor(val_meta_features, dtype=torch.float32).to(device)
    val_meta_labels = torch.tensor(val_meta_labels, dtype=torch.float32).unsqueeze(1).to(device)

    # Инициализация мета-модели
    meta_model = StackingMetaModel(num_models=len(models)).to(device)
    criterion = FocalLoss()
    optimizer = optim.AdamW(meta_model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # Обучение мета-модели
    best_auc = 0.0
    patience_counter = 0
    num_epochs = 50
    patience = 5

    for epoch in range(num_epochs):
        meta_model.train()
        optimizer.zero_grad()
        with autocast(device_type='cuda', dtype=torch.float16):
            outputs = meta_model(train_meta_features)
            loss = criterion(outputs, train_meta_labels)
        loss.backward()
        optimizer.step()

        # Валидация
        meta_model.eval()
        with torch.no_grad():
            with autocast(device_type='cuda', dtype=torch.float16):
                val_outputs = meta_model(val_meta_features)
                val_loss = criterion(val_outputs, val_meta_labels)

            train_probs = torch.sigmoid(outputs).cpu().numpy()
            val_probs = torch.sigmoid(val_outputs).cpu().numpy()

            train_auc = roc_auc_score(train_meta_labels.cpu().numpy(), train_probs)
            val_auc = roc_auc_score(val_meta_labels.cpu().numpy(), val_probs)

            # Вычисление точности
            train_preds = (train_probs >= 0.5).astype(int)
            val_preds = (val_probs >= 0.5).astype(int)
            train_accuracy = accuracy_score(train_meta_labels.cpu().numpy(), train_preds)
            val_accuracy = accuracy_score(val_meta_labels.cpu().numpy(), val_preds)
            train_precision = precision_score(train_meta_labels.cpu().numpy(), train_preds)
            val_precision = precision_score(val_meta_labels.cpu().numpy(), val_preds)
            train_recall = recall_score(train_meta_labels.cpu().numpy(), train_preds)
            val_recall = recall_score(val_meta_labels.cpu().numpy(), val_preds)
            train_f1 = f1_score(train_meta_labels.cpu().numpy(), train_preds)
            val_f1 = f1_score(val_meta_labels.cpu().numpy(), val_preds)

        print(f'Мета-модель - Epoch {epoch+1}/{num_epochs}, '
              f'Loss: {loss.item():.4f}, AUC: {train_auc:.4f}, Accuracy: {train_accuracy:.4f}, '
              f'Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}, '
              f'Val Loss: {val_loss.item():.4f}, Val AUC: {val_auc:.4f}, Val Accuracy: {val_accuracy:.4f}, '
              f'Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}')

        scheduler.step()

        # Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': meta_model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, checkpoint_dir=os.path.join(checkpoint_dir, 'meta_model'))
            patience_counter = 0
            print(f"Лучший результат для мета-модели достигнут в эпохе {epoch+1}, модель сохранена.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping для мета-модели")
                break
