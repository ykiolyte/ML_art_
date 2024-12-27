# src/validate.py

import torch
from torch.utils.data import DataLoader
from src.data_preparation import PS2Dataset, get_transforms
from src.model import PS2Net, StackingMetaModel
from src.utils import load_checkpoint
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def validate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")

    # Датасет и загрузчик данных
    val_dataset = PS2Dataset(data_dir='data/val', transform=get_transforms('val'), phase='val')
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=2)

    # Список моделей для ансамблирования
    model_names = ['efficientnet_v2_l', 'resnet50', 'convnext_base']
    models = []

    # Загрузка базовых моделей
    for model_name in model_names:
        model = PS2Net(model_name=model_name).to(device)
        load_checkpoint(model, checkpoint_dir=f'checkpoints/{model_name}')
        model.eval()
        models.append(model)

    # Загрузка мета-модели для стекинга
    meta_model = StackingMetaModel(num_models=len(models)).to(device)
    load_checkpoint(meta_model, checkpoint_dir='checkpoints/meta_model')
    meta_model.eval()

    all_labels = []
    all_preds_ensemble = []
    all_preds_stacking = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Валидация"):
            images = images.to(device, non_blocking=True)
            labels = labels.unsqueeze(1).float().to(device, non_blocking=True)
            batch_preds = []

            for model in models:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    probs = torch.sigmoid(outputs)
                batch_preds.append(probs.cpu().numpy())

            # Ансамблирование через усреднение
            ensemble_probs = np.mean(batch_preds, axis=0)
            all_preds_ensemble.extend(ensemble_probs.flatten())

            # Стекинг с мета-моделью
            meta_features = np.hstack(batch_preds)
            meta_features = torch.tensor(meta_features, dtype=torch.float32).to(device)
            with torch.cuda.amp.autocast():
                stacking_outputs = meta_model(meta_features)
                stacking_probs = torch.sigmoid(stacking_outputs).cpu().numpy()
            all_preds_stacking.extend(stacking_probs.flatten())

            all_labels.extend(labels.cpu().numpy().flatten())

    all_labels = np.array(all_labels)
    evaluate_predictions(all_labels, all_preds_ensemble, 'Ensemble', 'roc_curve_val_ensemble.png')
    evaluate_predictions(all_labels, all_preds_stacking, 'Stacking', 'roc_curve_val_stacking.png')

def evaluate_predictions(all_labels, all_preds, method_name, save_path):
    val_auc = roc_auc_score(all_labels, all_preds)
    val_preds = (all_preds >= 0.5).astype(int)
    val_precision = precision_score(all_labels, val_preds)
    val_recall = recall_score(all_labels, val_preds)
    val_f1 = f1_score(all_labels, val_preds)
    val_accuracy = accuracy_score(all_labels, val_preds)

    print(f'{method_name} - AUC: {val_auc:.4f}, '
          f'Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, '
          f'Recall: {val_recall:.4f}, F1-Score: {val_f1:.4f}')

    # Построение ROC-кривой
    fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {val_auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve ({method_name})')
    plt.legend(loc='lower right')
    plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    validate_model()
