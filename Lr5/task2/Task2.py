from PIL import Image, ImageEnhance
import numpy as np
from sklearn.cluster import KMeans
import cv2
import matplotlib.pyplot as plt



# Завантаження зображення
image = Image.open("Homer.png")
print(f"Оригінальний розмір зображення: {image.size}")

# Зменшення розміру, якщо потрібно
max_size = (300, 300)
if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
    image.thumbnail(max_size)
    print(f"Зменшено до: {image.size}")

# Збереження оригінального зображення для виводу
original_image = image.copy()

# Підвищення контрасту
enhancer = ImageEnhance.Contrast(image)
image = enhancer.enhance(2.0)  # Підвищення контрасту
enhanced_image = image.copy()  # Збереження покращеного зображення

# Конвертація в RGB
image = image.convert("RGB")

# Перетворення в масив пікселів
image_array = np.array(image)
pixel_data = image_array.reshape(-1, 3)  # Масив у форматі (N, 3)
print(f"Масив пікселів підготовлено: {pixel_data.shape}")

# Кластеризація методом k-середніх
k = 7  # Кількість кластерів
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(pixel_data)

cluster_centers = kmeans.cluster_centers_
labels = kmeans.labels_

print(f"Центри кластерів (основні кольори):\n{cluster_centers}")

# Створення кластеризованого зображення
clustered_data = cluster_centers[labels]
clustered_image = clustered_data.reshape(image_array.shape).astype(np.uint8)
print(f"Кластеризоване зображення створено з розміром: {clustered_image.shape}")

# Відображення результатів
plt.figure(figsize=(20, 10))

# Оригінальне зображення
plt.subplot(1, 4, 1)
plt.imshow(original_image)
plt.title("Оригінальне зображення")
plt.axis('off')

# Покращене зображення
plt.subplot(1, 4, 2)
plt.imshow(enhanced_image)
plt.title("Покращене зображення (контраст)")
plt.axis('off')

# Кластеризоване зображення
plt.subplot(1, 4, 3)
plt.imshow(clustered_image)
plt.title("Кластеризоване зображення")
plt.axis('off')

# Основні кольори
plt.subplot(1, 4, 4)
plt.title("Основні кольори")
for i, color in enumerate(cluster_centers):
    plt.fill_between([i, i + 1], 0, 1, color=color / 255.0)

plt.xlim(0, len(cluster_centers))
plt.axis('off')
plt.show()

# Виявлення контурів
gray_image = cv2.cvtColor(clustered_image, cv2.COLOR_RGB2GRAY)
edges = cv2.Canny(gray_image, 100, 200)

contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

contour_image = clustered_image.copy()
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

plt.figure(figsize=(10, 5))
plt.imshow(contour_image)
plt.title("Контури на зображенні")
plt.axis('off')
plt.show()
