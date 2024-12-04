import cv2
import numpy as np
from matplotlib import pyplot as plt


def load_and_preprocess_satellite_image(image_path):
    # Завантаження зображення
    image = cv2.imread(image_path)
    if image is None:
        print("Помилка: Неможливо завантажити зображення.")
        return None

    # Перетворення в Lab для покращення контрасту
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced_image = cv2.merge((l, a, b))
    enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_LAB2BGR)

    # Видалення шуму за допомогою Bilateral Filter
    filtered_image = cv2.bilateralFilter(enhanced_image, d=15, sigmaColor=75, sigmaSpace=75)

    # Показати результати
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Оригінальне зображення")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
    plt.title("Покращене зображення")
    plt.axis("off")

    plt.show()

    return filtered_image


def segment_water_and_objects(image):
    # Перетворення у формат HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Вибір діапазону кольорів води
    lower_water = np.array([70, 50, 50])
    upper_water = np.array([130, 255, 255])
    water_mask = cv2.inRange(hsv_image, lower_water, upper_water)

    # Інвертування маски, щоб отримати об'єкти
    object_mask = cv2.bitwise_not(water_mask)

    # Морфологічні операції для очищення
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    clean_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Показати результати
    plt.figure(figsize=(6, 6))
    plt.imshow(clean_mask, cmap="gray")
    plt.title("Маска об'єктів")
    plt.axis("off")
    plt.show()

    return clean_mask


def count_ships(mask, min_contour_area=500):
    # Знаходимо контури
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Фільтруємо контури за площею
    ship_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    # Візуалізація
    output_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(output_image, ship_contours, -1, (0, 255, 0), 2)

    # Показати результати
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Кількість кораблів: {len(ship_contours)}")
    plt.axis("off")
    plt.show()

    return len(ship_contours)



# Основний блок
image_path = 'Satellite_Image_Crimea_Sevastopol_Naval_Base.jpg'
preprocessed_image = load_and_preprocess_satellite_image(image_path)
object_mask = segment_water_and_objects(preprocessed_image)
ship_count = count_ships(object_mask, min_contour_area=800)
print(f"Кількість знайдених кораблів: {ship_count}")
