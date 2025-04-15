import os
import cv2
import mediapipe as mp
import numpy as np
import base64
import io
import time # Для возможной отладки времени выполнения
from flask import Flask, render_template, request

# --- Инициализация Flask ---
app = Flask(__name__)

# --- Настройка MediaPipe FaceLandmarker ---
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# --- Константы ---
# Путь к файлу модели FaceLandmarker (должен лежать рядом или в Dockerfile)
MODEL_PATH = 'face_landmarker_v2_with_blendshapes.task'
# Индексы ключевых точек
LEFT_TEMPLE = 139
RIGHT_TEMPLE = 368
LEFT_JAW = 58
RIGHT_JAW = 288
POINTS_TO_DRAW = [LEFT_TEMPLE, RIGHT_TEMPLE, LEFT_JAW, RIGHT_JAW]
# Канонические X-координаты для классификации (из стандартной модели MediaPipe)
# X отвечает за лево/право в локальной системе координат лица
CANONICAL_X = {
    LEFT_TEMPLE: -0.094716,
    RIGHT_TEMPLE: 0.094716,
    LEFT_JAW: -0.063613,
    RIGHT_JAW: 0.063613
}
# Порог для классификации (процент от ширины висков)
TOLERANCE_PERCENTAGE = 0.03 # 3% - можно тюнить

# --- Глобальная инициализация модели (для производительности) ---
face_landmarker_options = None
landmarker = None
try:
    # Убедимся, что файл модели существует перед инициализацией
    if os.path.exists(MODEL_PATH):
         face_landmarker_options = FaceLandmarkerOptions(
             base_options=BaseOptions(model_asset_path=MODEL_PATH),
             running_mode=VisionRunningMode.IMAGE, # Режим обработки изображений
             num_faces=1) # Ищем только одно лицо для оптимизации
         landmarker = FaceLandmarker.create_from_options(face_landmarker_options)
         print(f"FaceLandmarker модель '{MODEL_PATH}' успешно загружена.")
    else:
         print(f"ОШИБКА: Файл модели FaceLandmarker не найден по пути: {MODEL_PATH}")
         # Приложение может работать, но обработка не будет доступна
except Exception as e:
    print(f"ОШИБКА при инициализации FaceLandmarker: {e}")
    # Очищаем, чтобы не было попыток использовать неинициализированный объект
    face_landmarker_options = None
    landmarker = None


# --- Вспомогательные функции ---
def classify_jaw_width(canonical_x_coords, tolerance_percentage):
    """Классифицирует ширину челюсти на основе канонических X-координат."""
    try:
        X_temple_L = canonical_x_coords[LEFT_TEMPLE]
        X_temple_R = canonical_x_coords[RIGHT_TEMPLE]
        X_jaw_L = canonical_x_coords[LEFT_JAW]
        X_jaw_R = canonical_x_coords[RIGHT_JAW]

        inter_temple_width = abs(X_temple_R - X_temple_L)
        if inter_temple_width < 1e-6:
            return "Ошибка: Нулевая ширина висков (канон.)."

        tolerance = tolerance_percentage * inter_temple_width
        deviation_L = X_jaw_L - X_temple_L # Отклонение левой челюсти от левого виска
        deviation_R = X_jaw_R - X_temple_R # Отклонение правой челюсти от правого виска

        # Классификация сторон
        if deviation_L < -tolerance: class_L = "Широкая"
        elif deviation_L > tolerance: class_L = "Узкая"
        else: class_L = "Средняя"

        if deviation_R > tolerance: class_R = "Широкая"
        elif deviation_R < -tolerance: class_R = "Узкая"
        else: class_R = "Средняя"

        # Итоговая классификация
        if class_L == "Широкая" and class_R == "Широкая": result = "Широкая"
        elif class_L == "Узкая" and class_R == "Узкая": result = "Узкая"
        else: result = "Средняя"

        print(f"Классификация: Шир.висков={inter_temple_width:.4f}, Порг={tolerance:.4f}, Откл.L={deviation_L:.4f} ({class_L}), Откл.R={deviation_R:.4f} ({class_R}) -> ИТОГ={result}")
        return result

    except KeyError as e:
        print(f"Ошибка классификации: Отсутствует каноническая координата для точки {e}")
        return "Ошибка: Нет данных для точки."
    except Exception as e:
        print(f"Неизвестная ошибка классификации: {e}")
        return f"Ошибка классификации: {e}"

def draw_on_image(image_np, detection_result):
    """Рисует сетку и точки на изображении NumPy."""
    annotated_image = image_np.copy()
    image_height, image_width, _ = annotated_image.shape

    if detection_result.face_landmarks:
        face_landmarks_list = detection_result.face_landmarks[0] # Берем первое лицо

        # --- Рисуем сетку ---
        # Создаем нужный формат для draw_landmarks
        from mediapipe.framework.formats import landmark_pb2
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks_list
        ])
        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION, # Используем стандартные связи
            landmark_drawing_spec=None, # Не рисуем все 478 точек сетки
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())

        # --- Рисуем ключевые точки ---
        for idx in POINTS_TO_DRAW:
            try:
                lm = face_landmarks_list[idx]
                x_px = int(lm.x * image_width)
                y_px = int(lm.y * image_height)
                cv2.circle(annotated_image, (x_px, y_px), radius=3, color=(0, 0, 255), thickness=-1) # Красные точки
            except IndexError:
                print(f"Предупреждение: Не удалось нарисовать точку с индексом {idx}")

    return annotated_image


# --- Основная функция обработки ---
def process_image_and_classify(image_bytes):
    """
    Главная функция: декодирует, запускает FaceLandmarker, классифицирует, рисует.
    Возвращает байты обработанного изображения и строку с результатом/ошибкой.
    """
    start_time = time.time()
    classification_result = "Не удалось обработать."
    processed_image_bytes = None

    # Проверяем, инициализирована ли модель
    if landmarker is None:
        print("ОШИБКА: FaceLandmarker не инициализирован.")
        return None, "Ошибка сервера: Модель не загружена."

    image_np = None
    mp_image = None
    detection_result = None
    annotated_image = None

    try:
        # 1. Декодируем изображение
        nparr = np.frombuffer(image_bytes, np.uint8)
        image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image_np is None:
            return None, "Ошибка: Не удалось декодировать изображение."

        # 2. Создаем формат MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)

        # 3. Запускаем детекцию
        detection_result = landmarker.detect(mp_image)

        # 4. Классифицируем (на основе канонических координат)
        if not detection_result or not detection_result.face_landmarks:
             classification_result = "Лицо не найдено."
             # В этом случае просто вернем картинку с надписью (нарисуем позже)
        else:
             classification_result = classify_jaw_width(CANONICAL_X, TOLERANCE_PERCENTAGE)

        # 5. Рисуем результат на изображении
        # Рисуем всегда, даже если лицо не найдено (чтобы вернуть картинку)
        annotated_image = draw_on_image(image_np, detection_result)
        # Если лицо не найдено, добавляем текст
        if classification_result == "Лицо не найдено.":
             cv2.putText(annotated_image, "Face not detected", (50, 50),
                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 6. Кодируем результат в JPEG байты
        is_success, buffer = cv2.imencode(".jpg", annotated_image)
        if is_success:
            processed_image_bytes = buffer.tobytes()
        else:
            classification_result = "Ошибка: Не удалось закодировать результат."
            processed_image_bytes = None # Явно обнуляем

    except Exception as e:
        print(f"Ошибка в process_image_and_classify: {e}")
        # Пытаемся вернуть сообщение об ошибке, если возможно
        if classification_result == "Не удалось обработать.": # Если еще не было специфической ошибки
             classification_result = f"Внутренняя ошибка: {e}"
        processed_image_bytes = None # Явно обнуляем

    finally:
        # 7. Очистка памяти (на всякий случай)
        del image_np
        del mp_image
        del detection_result
        del annotated_image
        end_time = time.time()
        print(f"Время обработки: {end_time - start_time:.4f} сек.")

    return processed_image_bytes, classification_result


# --- Маршруты Flask ---
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', error_message="Файл не выбран.")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error_message="Файл не выбран.")

    if file:
        original_image_bytes = file.read()
        processed_image_bytes, status_message = process_image_and_classify(original_image_bytes)

        # Освобождаем память от исходных байт СРАЗУ после обработки
        del original_image_bytes

        processed_image_b64 = None
        if processed_image_bytes:
             processed_image_b64 = base64.b64encode(processed_image_bytes).decode('utf-8')
             del processed_image_bytes # Освобождаем и эти байты после кодирования в b64

        return render_template('index.html',
                               status_message=status_message,
                               processed_image_data=processed_image_b64)

    return render_template('index.html', error_message="Ошибка при загрузке файла.")

# --- Запуск приложения ---
if __name__ == '__main__':
    # Убедимся, что модель загрузилась перед запуском сервера
    if landmarker:
        print("Запуск Flask сервера...")
        # Gunicorn будет управлять этим в продакшене
        app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=False)
    else:
        print("ОШИБКА: Не удалось загрузить модель FaceLandmarker. Сервер не запущен.")
