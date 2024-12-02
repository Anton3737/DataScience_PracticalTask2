import numpy as np

def generate_error(mean, std_dev, num_samples):
    """Генеруємо випадкову похибку з нормального розподілу."""
    return np.random.normal(mean, std_dev, num_samples)
