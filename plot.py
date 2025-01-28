import matplotlib.pyplot as plt
import numpy as np

# Данные
experiments = ['UNET_MSE_FACTOR_00', 'UNET_MSE_FACTOR_010', 'UNET_MSE_FACTOR_03', 'UNET_MSE_FACTOR_06', 'UNET_MSE_FACTOR_08', ]
tpr_values = [0.9612, 0.9457, 0.9922, 0.9457, 0.9380, ]
tnr_values = [0.8488, 0.8628, 0.8322, 0.8379, 0.8161, ]

experiments += ['UNET_MSE_FACTOR_00_VGG', 'UNET_MSE_FACTOR_03_VGG', 'UNET_MSE_FACTOR_06_VGG']
tpr_values += [0.9767, 0.9612, 0.9535]
tnr_values += [0.7850, 0.8357, 0.8284]

# Устанавливаем ширину столбцов
bar_width = 0.35

# Определяем позиции для столбцов
index = np.arange(len(experiments))

# Создаём график
fig, ax = plt.subplots(figsize=(10, 6))

# Столбцы для TPR и TNR
bar1 = ax.bar(index, tpr_values, bar_width, label='TPR', color='b')
bar2 = ax.bar(index + bar_width, tnr_values, bar_width, label='TNR', color='g')

# Добавление подписей и заголовка
ax.set_xlabel('Эксперименты')
ax.set_ylabel('Значения')
ax.set_title('Сравнение TPR и TNR для порога TPR 95')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(experiments, rotation=45, ha='right')
ax.legend()

ax.set_ylim(0.8, 1)

# Показать график
plt.tight_layout()
plt.show()