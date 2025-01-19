# processing_and_generating_images_course
Студент: Запольский Максим Михайлович (ВШЭ, МОАД)
## HW1: Классические задачи Computer Vision

Будем решать задачу классификации. В качестве датасета возьмем FASTION-MNIST

### Эксперимент

В качестве архитектуры будем использовать ResNet

Параметры обучения:

```python
epochs=10
batch_size=64
lr=0.001
```

В конце выберем те веса, которые минимизировали val_loss и проверим метрики на test датасете, в итоге получаем:

```
Test set: Average loss: 0.0045 
Accuracy: 0.9255
Precision: 0.9250980892600308
Recall: 0.9255
F1_score: 0.9250024650416366
```

# Эксперимент 2

Попробуем обучить vision transformer. Как и в прошлом эксперименте выберем в итоге ту модель, у которой будет лучший результат на валидации.

Параметры:
```python
epochs=10
batch_size=64
lr=1e-5
img_size=28,
patch_size=4,
in_channels=1,
embed_dim=128,
num_heads=16,
num_layers=6,
mlp_hidden_dim=256,
num_classes=10,
drop_prob=0.1
```
 
Результат:

```
Test set: Average loss: 0.0062 
 Accuracy: 0.8607
 Precision: 0.8635000508760593
 Recall: 0.8607
 F1_score: 0.8611873647712848
```


Попробуем сделать сеть больше:
```python
epochs=10
batch_size=64
lr=1e-5
img_size=28,
patch_size=4,
in_channels=1,
embed_dim=512,
num_heads=32,
num_layers=12,
mlp_hidden_dim=512,
num_classes=10,
drop_prob=0.1
```

Результат:

```
Test set: Average loss: 0.0077 
 Accuracy: 0.8579
 Precision: 0.8589171145320069
 Recall: 0.8579
 F1_score: 0.8563031840093652
```

Ожидаемо результат хуже, так как данных для обучения трансформера у нас мало. Вероятно, если взять предобученную модель на большом сете, результаты были бы значительно лучше.

Видно, что сверточная архитектура на маленьком датасете лучше выучивается извлечению признаков.

Также увеличение размера сети привело к переобучению, что видно на графике:

![img.png](img.png)

[Логи wandb](https://wandb.ai/revelia/HW1-fashion-mnist/workspace?nw=nwuserrevelia)



