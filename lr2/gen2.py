from error_gen import generate_error
from process_data import generate_process_data
from save_to_file import save_data_to_csv
import pandas as pd

mean = 10
std_dev = 10
num_samples = 1000
a, b, c = 1, 0.5, 2

# Генерація даних
x, process_data = generate_process_data(a, b, c, num_samples)
measurement_error = generate_error(mean, std_dev, num_samples)
experimental_data = process_data + measurement_error

data_df = pd.DataFrame({
    "x": x,
    "process_data": process_data,
    "measurement_error": measurement_error,
    "experimental_data": experimental_data
})

# Збереження у CSV
save_data_to_csv(data_df, "data_with_anomalies.csv")
