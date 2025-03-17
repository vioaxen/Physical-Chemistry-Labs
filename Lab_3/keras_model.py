import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")

# Загрузка модели
model = tf.keras.models.load_model('Lab3_ML_Propane.keras')

# Загрузка данных
data = pd.read_csv("data.csv")

temperatures = data["Temperature"].values
pressures = data["Pressure"].values
densities = data["Density"].values

# Границы нормализации
minsX = np.array([230.74, 1.0])
maxsX = np.array([1000, 10.0])
minsy = np.array([0.5302])
maxsy = np.array([21.67])

def normalize(X, mins, maxs):
    return (X - mins) / (maxs - mins) * 0.9 + 0.1

def denormalize(X, mins, maxs):
    return ((X - 0.1) / 0.9) * (maxs - mins) + mins

# Уникальные значения давления
unique_pressures_all = np.unique(pressures)

plt.figure(figsize=(9, 6))
colors = sns.color_palette("coolwarm", len(unique_pressures_all))

for i, P in enumerate(unique_pressures_all):
    mask = pressures == P
    T_exp = temperatures[mask]
    rho_exp = densities[mask]

    T_pred = np.linspace(min(T_exp), max(T_exp), 100)
    P_pred = np.full_like(T_pred, P)
    X_pred = np.column_stack((T_pred, P_pred))
    X_pred_norm = normalize(X_pred, minsX, maxsX)
    y_pred_norm = model.predict(X_pred_norm)
    y_pred = denormalize(y_pred_norm, minsy, maxsy)

    plt.scatter(T_exp, rho_exp, color=colors[i], label=f'P={P} бар (эксп.)', edgecolors='black', s=60, alpha=0.8)
    plt.plot(T_pred, y_pred, linestyle='-', color=colors[i], alpha=0.8, label=f'P={P} бар (модель)')

# Предсказание для T=330K, P=1 бар
X_new = np.array([[330, 1]])
X_new_norm = normalize(X_new, minsX, maxsX)
y_new_norm = model.predict(X_new_norm)
y_new = denormalize(y_new_norm, minsy, maxsy)
plt.scatter(330, y_new[0, 0], color='red', s=180, marker='*', label=f'Предсказание (T=330K, P=1 бар)', edgecolors='black')

plt.xlabel('Температура [K]', fontsize=14)
plt.ylabel('Плотность [кг/м³]', fontsize=14)
plt.title('Зависимость плотности пропана от температуры', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend(title="Давление", fontsize=12, loc='lower left')

save_path = os.path.join('plot_propane.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"Предсказанная плотность пропана при T=330K, P=1 бар: {y_new[0,0]:.2f} кг/м³")

data = pd.read_csv("fixed_data.csv")

temperatures = data["Temperature"].values
pressures = data["Pressure"].values
densities = data["Density"].values

fixed_pressures = [1.0, 10.0]
for fixed_P in fixed_pressures:
    plt.figure(figsize=(9, 6))
    mask = pressures == fixed_P
    T_exp = temperatures[mask]
    rho_exp = densities[mask]
    print(*T_exp)
    T_pred = np.linspace(min(T_exp), max(T_exp), 100)
    P_pred = np.full_like(T_pred, fixed_P)
    X_pred = np.column_stack((T_pred, P_pred))
    X_pred_norm = normalize(X_pred, minsX, maxsX)
    y_pred_norm = model.predict(X_pred_norm)
    y_pred = denormalize(y_pred_norm, minsy, maxsy)

    plt.scatter(T_exp, rho_exp, color='blue', label='Экспериментальные данные', edgecolors='black', s=60, alpha=0.8)
    plt.plot(T_pred, y_pred, linestyle='-', color='red', label='Предсказание модели', alpha=0.8)

    plt.xlabel('Температура [K]', fontsize=14)
    plt.ylabel('Плотность [кг/м³]', fontsize=14)
    plt.title(f'Зависимость плотности пропана от температуры\nпри P={fixed_P} бар', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(fontsize=12)

    save_path = os.path.join(f'plot_P_{int(fixed_P)}_bar.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

print(f"Предсказанная плотность пропана при T=330K, P=1 бар: {y_new[0,0]:.2f} кг/м³")
