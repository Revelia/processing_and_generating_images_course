# processing_and_generating_images_course

## Запольский Максим Михайлович

## Info

- Точка входа `main.py`
- Графики по экспериментам хранятся в папках `~/experiment_name`
- [логи wandb](https://wandb.ai/revelia/HW2-img-processing-course)



## Предобработка данных

Заранее вычислим mean и std на нашем датасете:

```python
MEAN = [0.42343804240226746, 0.5342181324958801, 0.4620889723300934]
STD = [0.04519784078001976, 0.05054565891623497, 0.046623989939689636]
```

При обучении будем использовать следующие трансформации:
```python
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
```

Отражения для аугментации, Resize для приведения картинок к одному размеру (они имеют слегка разное разрешение в датасете)

Другие аргументации, кажется, смысла не имеют

## Архитектура

Будем использовать U-net

![img.png](img.png)

За кадром были попытки использовать

- VAE
- ViT

Но VAE делает результат слишком мыльным, а для ViT недостаточно данных.

## Обучение

### Loss
В качестве loss функции будем среднее

$$loss = \lambda \cdot loss_{MSE} + (1 - \lambda )\cdot loss_{perceptual} $$

Где $loss_{perceptual}$ вычисляется как 

$$loss_{perceptual} = MSE[AlexNet(original), AlexNet(reconstruction)]$$

### Данные

В самом начале поделим train на train/val датасеты с соотношением 0.9/0.1.

### Пайплайн

Будем делать следующее:

1. Учимся на train, следя за метриками на валидации, чтобы не допустить переобучения
2. По val/proliv подбираем пороговое значение на основе: f1, tpr=0.95, tnr=0.95
3. Для полученных пороговых значений вычисляем метрики на тесте.

## Эксперименты

### MSE_FACTOR

Проведем серию экспериментов, варьируя MSE_FACTOR:

```python
        {"experiment_name": "UNET_MSE_FACTOR_00", "mse_factor": 0.0},
        {"experiment_name": "UNET_MSE_FACTOR_03", "mse_factor": 0.3},
        {"experiment_name": "UNET_MSE_FACTOR_06", "mse_factor": 0.6},
        {"experiment_name": "UNET_MSE_FACTOR_08", "mse_factor": 0.8},
        {"experiment_name": "UNET_MSE_FACTOR_10", "mse_factor": 0.10},
```

Графики экспериментов и пример реконструкции будут лежать в соответствующих папках, логи обучения в wandb. Приведем тут только краткие результаты:

```
==========
Имя эксперимента: UNET_MSE_FACTOR_00

----------
Info: Best F1:
Threshold: 1.3541354135413541:
True Positives: 126, False Positives: 618
True Negatives: 3047, False Negatives: 3
True Positive Rate (TPR): 0.9767
True Negative Rate (TNR): 0.8314
----------
Info: TPR 95:
Threshold: 1.5131513151315132:
True Positives: 124, False Positives: 554
True Negatives: 3111, False Negatives: 5
True Positive Rate (TPR): 0.9612
True Negative Rate (TNR): 0.8488
----------
Info: TNR 95:
Threshold: 0.725072507250725:
True Positives: 129, False Positives: 1653
True Negatives: 2012, False Negatives: 0
True Positive Rate (TPR): 1.0000
True Negative Rate (TNR): 0.5490

==========
Имя эксперимента: UNET_MSE_FACTOR_03
----------
Info: Best F1:
Threshold: 1.5501550155015502:
True Positives: 128, False Positives: 647
True Negatives: 3018, False Negatives: 1
True Positive Rate (TPR): 0.9922
True Negative Rate (TNR): 0.8235
----------
Info: TPR 95:
Threshold: 1.6061606160616062:
True Positives: 128, False Positives: 615
True Negatives: 3050, False Negatives: 1
True Positive Rate (TPR): 0.9922
True Negative Rate (TNR): 0.8322
----------
Info: TNR 95:
Threshold: 1.0891089108910892:
True Positives: 129, False Positives: 1257
True Negatives: 2408, False Negatives: 0
True Positive Rate (TPR): 1.0000
True Negative Rate (TNR): 0.6570
==========
Имя эксперимента: UNET_MSE_FACTOR_06
----------
Info: Best F1:
Threshold: 1.7431743174317431:
True Positives: 128, False Positives: 647
True Negatives: 3018, False Negatives: 1
True Positive Rate (TPR): 0.9922
True Negative Rate (TNR): 0.8235
----------
----------
Info: TPR 95:
Threshold: 1.8961896189618963:
True Positives: 122, False Positives: 594
True Negatives: 3071, False Negatives: 7
True Positive Rate (TPR): 0.9457
True Negative Rate (TNR): 0.8379
----------
----------
Info: TNR 95:
Threshold: 1.3671367136713672:
True Positives: 129, False Positives: 1025
True Negatives: 2640, False Negatives: 0
True Positive Rate (TPR): 1.0000
True Negative Rate (TNR): 0.7203
----------
==========
Имя эксперимента: UNET_MSE_FACTOR_08

==========
Имя эксперимента: UNET_MSE_FACTOR_10

==========

```



