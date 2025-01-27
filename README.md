# processing_and_generating_images_course

## Запольский Максим Михайлович

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

В качестве loss функции будем среднее

$$loss = \lambda \cdot loss_{MSE} + (1 - \lambda )\cdot loss_{perceptual} $$

Где $loss_{perceptual}$ вычисляется как 

$$loss_{perceptual} = MSE[AlexNet(original), AlexNet(reconstruction)]$$

