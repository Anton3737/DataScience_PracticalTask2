import psutil
import platform
import subprocess
import matplotlib.pyplot as plt
import time

# 1. Збір даних про систему
def get_system_metrics():
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_percent = psutil.virtual_memory().percent
    disk_percent = psutil.disk_usage('/').percent
    cpu_temp = None

    try:
        cpu_temp = psutil.sensors_temperatures().get('coretemp', [])[0].current
    except Exception:
        cpu_temp = None

    os_info = platform.uname()
    mac_version = None
    if platform.system() == 'Darwin':
        mac_version = platform.mac_ver()[0]

    return cpu_percent, memory_percent, disk_percent, cpu_temp, os_info, mac_version

# 2. Отримання інформації про GPU
def get_gpu_info():
    gpu_info = "No GPU found or unable to retrieve GPU information."
    try:
        gpu_output = subprocess.check_output("system_profiler SPDisplaysDataType", shell=True).decode()
        if "Graphics" in gpu_output:
            gpu_info = gpu_output.split("Graphics")[1].split("\n")[0].strip()
    except Exception:
        pass
    return gpu_info

# 3. Функція багатокритеріальної оптимізації
def multi_criteria_optimization(cpu_percent, memory_percent):
    try:
        speed = 1 / (cpu_percent * memory_percent / 100) if cpu_percent * memory_percent > 0 else 0
        cpu_usage = -cpu_percent
        reliability = 1 - (cpu_percent + memory_percent) / 200
        energy_efficiency = 1 / (cpu_percent * memory_percent / 100) if cpu_percent * memory_percent > 0 else 0
    except ZeroDivisionError:
        speed = cpu_usage = reliability = energy_efficiency = 0
    return speed, cpu_usage, reliability, energy_efficiency

# 4. Побудова графіка
def plot_performance(cpu_data, memory_data, duration=30):
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(cpu_data, label='CPU Usage (%)', color='red')
    plt.title("CPU Usage Over Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("CPU Usage (%)")
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(memory_data, label='Memory Usage (%)', color='blue')
    plt.title("Memory Usage Over Time")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Memory Usage (%)")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

# 5. Основна функція
def main():
    duration = 30
    cpu_data, memory_data = [], []
    start_time = time.time()

    # Збір даних протягом 30 секунд
    while time.time() - start_time < duration:
        cpu_percent, memory_percent, _, _, _, _ = get_system_metrics()
        cpu_data.append(cpu_percent)
        memory_data.append(memory_percent)
        time.sleep(1)

    # Отримуємо останні дані для виводу
    cpu_percent, memory_percent, disk_percent, cpu_temp, os_info, mac_version = get_system_metrics()
    gpu_info = get_gpu_info()

    # Виводимо інформацію про систему
    print(f"Використання CPU: {cpu_percent:.2f}%")
    print(f"Використання пам'яті: {memory_percent:.2f}%")
    print(f"Використання диска: {disk_percent:.2f}%")
    if cpu_temp:
        print(f"Температура CPU: {cpu_temp:.2f}°C")

    print("\nОпераційна система:")
    if mac_version:
        print(f"  Версія macOS: {mac_version}")
    else:
        print(f"  OS: {os_info.system}")
        print(f"  Версія: {os_info.release}")
        print(f"  Повна версія: {os_info.version}")

    print("\nІнформація про апаратне забезпечення:")
    print(f"  Машина: {os_info.machine}")
    print(f"  Процесор: {os_info.processor}")
    print(f"GPU: {gpu_info}")

    # Багатокритеріальна оптимізація
    optimal_values = multi_criteria_optimization(cpu_percent, memory_percent)

    print("\nОптимізовані значення для системи:")
    print(f"Швидкість: {optimal_values[0]:.2f}")
    print(f"Використання CPU: {optimal_values[1]:.2f}")
    print(f"Надійність: {optimal_values[2]:.2f}")
    print(f"Енергоефективність: {optimal_values[3]:.2f}")

    # Побудова графіка
    plot_performance(cpu_data, memory_data, duration)

if __name__ == "__main__":
    main()
