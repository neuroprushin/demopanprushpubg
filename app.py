import os
import cv2
import mediapipe as mp
import numpy as np
import base64
import io
from flask import Flask, render_template, request
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- Инициализация Flask ---
app = Flask(__name__)

BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1
)
face_landmarker = FaceLandmarker.create_from_options(options)

# --- Настройка MediaPipe ---
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- Индексы ключевых точек (обновленные) ---
LEFT_TEMPLE = 139
RIGHT_TEMPLE = 368
LEFT_JAW = 58
RIGHT_JAW = 288
# Точки для рисования (можно добавить еще, если нужно)
POINTS_TO_DRAW = [LEFT_TEMPLE, RIGHT_TEMPLE, LEFT_JAW, RIGHT_JAW]

# --- Константы для классификации ---
# Процент от ширины между висками для определения "средней" зоны
TOLERANCE_PERCENTAGE = 0.03 # 3% - можно тюнить

# --- Функция обработки изображения ---
def process_image_and_classify(image_bytes):
    """
    Обрабатывает изображение, находит лицо, классифицирует ширину челюсти (3 ступени)
    и возвращает байты обработанного изображения и результат классификации.
    """
    classification_result = "Ошибка: Не удалось обработать."
    processed_image_bytes = None

    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return None, "Ошибка: Не удалось декодировать изображение."

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_height, image_width, _ = image.shape
        annotated_image = image.copy()

        # Инициализация Face Mesh и Face Geometry
        # estimate_head_pose=True важно для получения метрических координат
        with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True, # Оставляем для точности сетки
                min_detection_confidence=0.5) as face_mesh, \
             mp_face_geometry.FaceGeometry(estimate_head_pose=True) as face_geometry:

            results_mesh = face_mesh.process(image_rgb)

            if not results_mesh.multi_face_landmarks:
                classification_result = "Лицо не найдено."
                # Кодируем оригинал или картинку с надписью для вывода
                cv2.putText(annotated_image, "Face not detected", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                is_success, buffer = cv2.imencode(".jpg", annotated_image)
                processed_image_bytes = buffer.tobytes() if is_success else None
                return processed_image_bytes, classification_result

            # --- Обработка найденного лица ---
            face_landmarks = results_mesh.multi_face_landmarks[0]

            # Получаем метрические 3D-координаты
            try:
                # Обрати внимание: process ожидает форму изображения (height, width, channels)
                # и список лицевых ориентиров
                face_geometry_result = face_geometry.process(
                    image_shape=image_rgb.shape,
                    face_landmarks_proto=face_landmarks # Передаем объект из Face Mesh
                )
                # Получаем массив NumPy (478, 3) в метрах
                metric_landmarks = face_geometry_result.get_metric_landmarks()
            except Exception as e:
                 print(f"Ошибка при получении метрических координат: {e}")
                 # Можно вернуть ошибку или продолжить без классификации
                 classification_result = "Ошибка геометрии лица."
                 metric_landmarks = None # Ставим флаг, чтобы пропустить классификацию


            # --- Классификация ширины челюсти (если метрические точки получены) ---
            if metric_landmarks is not None:
                try:
                    # Извлекаем X-координаты (предполагаем, что X - это лево/право в локальной системе)
                    # Индексация [index][0] - берем X координату
                    X_temple_L = metric_landmarks[LEFT_TEMPLE][0]
                    X_temple_R = metric_landmarks[RIGHT_TEMPLE][0]
                    X_jaw_L = metric_landmarks[LEFT_JAW][0]
                    X_jaw_R = metric_landmarks[RIGHT_JAW][0]

                    # Считаем ширину и порог
                    inter_temple_width = abs(X_temple_R - X_temple_L)
                    if inter_temple_width < 1e-6: # Защита от деления на ноль
                        classification_result = "Ошибка: Нулевая ширина висков."
                    else:
                        tolerance = TOLERANCE_PERCENTAGE * inter_temple_width

                        # Считаем отклонения
                        deviation_L = X_jaw_L - X_temple_L
                        deviation_R = X_jaw_R - X_temple_R

                        # Классифицируем стороны
                        if deviation_L < -tolerance: class_L = "Широкая"
                        elif deviation_L > tolerance: class_L = "Узкая"
                        else: class_L = "Средняя"

                        if deviation_R > tolerance: class_R = "Широкая"
                        elif deviation_R < -tolerance: class_R = "Узкая"
                        else: class_R = "Средняя"

                        # Итоговая классификация
                        if class_L == "Широкая" and class_R == "Широкая":
                            classification_result = "Широкая"
                        elif class_L == "Узкая" and class_R == "Узкая":
                            classification_result = "Узкая"
                        else:
                            classification_result = "Средняя"

                        # Добавим отладочную инфу (можно убрать потом)
                        print(f"Данные классификации:")
                        print(f"  Ширина висков (X): {inter_temple_width:.4f}")
                        print(f"  Порог (X): {tolerance:.4f}")
                        print(f"  Отклонение L: {deviation_L:.4f} ({class_L})")
                        print(f"  Отклонение R: {deviation_R:.4f} ({class_R})")
                        print(f"  ИТОГ: {classification_result}")

                except IndexError:
                    classification_result = "Ошибка: Неверный индекс точки."
                except Exception as e:
                    classification_result = f"Ошибка классификации: {e}"
            else:
                 # Если метрические точки не получены, classification_result уже содержит ошибку
                 pass


            # --- Рисование на изображении ---
            # Рисуем сетку
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

            # Рисуем ключевые точки (виски и челюсть)
            landmarks_px = {}
            for idx in POINTS_TO_DRAW:
                 lm = face_landmarks.landmark[idx]
                 x_px = int(lm.x * image_width)
                 y_px = int(lm.y * image_height)
                 landmarks_px[idx] = (x_px, y_px)
                 # Рисуем точку
                 cv2.circle(annotated_image, (x_px, y_px), radius=3, color=(0, 0, 255), thickness=-1) # Красные точки

            # Можно добавить линии для наглядности, если хочешь
            # cv2.line(annotated_image, landmarks_px[LEFT_TEMPLE], landmarks_px[LEFT_JAW], (255,0,0), 1) # Синяя линия слева
            # cv2.line(annotated_image, landmarks_px[RIGHT_TEMPLE], landmarks_px[RIGHT_JAW], (255,0,0), 1) # Синяя линия справа

            # Кодируем результат в байты
            is_success, buffer = cv2.imencode(".jpg", annotated_image)
            if not is_success:
                # Если кодирование не удалось, результат уже содержит ошибку
                processed_image_bytes = None
                classification_result = "Ошибка: Не удалось закодировать результат."
            else:
                processed_image_bytes = buffer.tobytes()

    except Exception as e:
        print(f"Критическая ошибка в process_image_and_classify: {e}")
        classification_result = f"Внутренняя ошибка сервера: {e}"
        processed_image_bytes = None # Явно обнуляем

    return processed_image_bytes, classification_result


# --- Маршруты Flask ---

@app.route('/', methods=['GET'])
def index():
    """ Отображает главную страницу """
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """ Обрабатывает загрузку файла и показывает результат """
    if 'file' not in request.files:
        return render_template('index.html', error_message="Файл не выбран.")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error_message="Файл не выбран.")

    if file:
        original_image_bytes = file.read()
        processed_image_bytes, status_message = process_image_and_classify(original_image_bytes)

        processed_image_b64 = None
        if processed_image_bytes:
             processed_image_b64 = base64.b64encode(processed_image_bytes).decode('utf-8')
        else:
            # Если обработка не удалась, передаем сообщение об ошибке
            pass # status_message уже должен содержать ошибку

        # Передаем результат классификации и картинку (или сообщение об ошибке)
        return render_template('index.html',
                               status_message=status_message,
                               processed_image_data=processed_image_b64) # Убрали original_image_data

    return render_template('index.html', error_message="Ошибка при загрузке файла.")


# --- Запуск приложения ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=False)
