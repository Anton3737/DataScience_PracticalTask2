import numpy as np
import matplotlib.pyplot as plt

# Вхідні дані (Input dataset)
X = np.array([
    [0, 1, 1],
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 1],
    [0, 0, 1]
])

# Вихідні дані (Output dataset)
y = np.array([[0], [0], [0], [1], [1]])

# Випадкова ініціалізація ваг
np.random.seed(76)
weights = np.random.rand(3, 1) - 0.5  # Випадкові ваги (від -0.5 до 0.5)
bias = np.random.rand(1) - 0.5  # Випадковий bias
learning_rate = 0.1  # Швидкість навчання

# Сигмоїдна функція
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Похідна сигмоїди
def sigmoid_derivative(x):
    return x * (1 - x)

# Навчання нейронної мережі
epochs = 100000  # Кількість ітерацій
errors = []  # Для збереження похибки на кожній епосі

for epoch in range(epochs):
    # Прямий прохід
    z = np.dot(X, weights) + bias  # Зважена сума
    output = sigmoid(z)  # Результат активації

    # Похибка
    error = y - output
    errors.append(np.mean(np.abs(error)))  # Середня абсолютна похибка

    # Зворотне поширення
    d_output = error * sigmoid_derivative(output)
    weights += np.dot(X.T, d_output) * learning_rate
    bias += np.sum(d_output) * learning_rate

# Графік зміни похибки
plt.plot(errors)
plt.title("Зміна похибки під час навчання")
plt.xlabel("Епоха")
plt.ylabel("Середня абсолютна похибка")
plt.show()

# Перевірка працездатності
print("Остаточні ваги:", weights)
print("Остаточний bias:", bias)

# Передбачення
predictions = sigmoid(np.dot(X, weights) + bias)
print("\nПередбачення:")
print(predictions)

# Округлення результатів
rounded_predictions = np.round(predictions)
print("\nОкруглені передбачення:")
print(rounded_predictions)

# Оцінка точності
accuracy = np.mean(rounded_predictions == y) * 100
print(f"\nТочність моделі: {accuracy:.2f}%")
