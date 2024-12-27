# main.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Отключаем логи TensorFlow

import argparse
from src.train import train_model
from src.test import test_model
from src.validate import validate_model

def main():
    parser = argparse.ArgumentParser(description='Запуск модели PS2Net')
    parser.add_argument('--phase', type=str, default='train', choices=['train', 'test', 'validate', 'all'],
                        help='Фаза работы: train, test, validate, all')
    args = parser.parse_args()

    if args.phase == 'train':
        print("Запуск обучения модели...")
        train_model()
    elif args.phase == 'test':
        print("Запуск тестирования модели...")
        test_model()
    elif args.phase == 'validate':
        print("Запуск валидации модели...")
        validate_model()
    elif args.phase == 'all':
        print("Запуск обучения модели...")
        train_model()
        print("Запуск валидации модели...")
        validate_model()
        print("Запуск тестирования модели...")
        test_model()
    else:
        print("Неизвестная фаза работы. Используйте --help для просмотра доступных опций.")

if __name__ == '__main__':
    main()
