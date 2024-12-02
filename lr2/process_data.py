import numpy as np

def generate_process_data(a, b, c, num_samples):
    """Генеруємо дані процесу на основі квадратичного рівняння."""
    x = np.linspace(-10, 10, num_samples)
    process_data = a * x**2 + b * x + c
    return x, process_data
