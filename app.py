import os
import cv2
import mediapipe as mp
import numpy as np
import base64
import io
from flask import Flask, render_template, request, send_from_directory

# --- Инициализация Flask ---
app = Flask(__name__)
# Указываем папку для загрузок (если нужно сохранять временно)
# UPLOAD_FOLDER = 'uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

# --- Настройка MediaPipe (как раньше) ---
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- Индексы ключевых точек (как раньше) ---
LEFT_EYE_INNER_CORNER = 33
LEFT_EYE_OUTER_CORNER = 133
RIGHT_EYE_INNER_CORNER = 362
RIGHT_EYE_OUTER_CORNER = 263
LEFT_JAW_ANGLE = 172
RIGHT_JAW_ANGLE = 397
LEFT_TEMPLE = 139
RIGHT_TEMPLE = 368
NOSE_BRIDGE = 9
NOSE_BASE = 2

# --- Функция обработки изображения (адаптированная для веб) ---
def process_face_mesh(image_bytes):
    """
    Обрабатывает изображение из байтов, рисует сетку и точки,
    возвращает байты обработанного изображения и статус.
    """
    try:
        # Декодируем изображение из байтов
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return None, "Ошибка: Не удалось декодировать изображение."

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_height, image_width, _ = image.shape
        annotated_image = image.copy() # Копия для рисования

        status_message = "Лицо не найдено." # Сообщение по умолчанию

        # Инициализация и запуск Face Mesh
        with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5) as face_mesh:

            results = face_mesh.process(image_rgb)

            if results.multi_face_landmarks:
                status_message = "Лицо найдено, точки построены."
                face_landmarks = results.multi_face_landmarks[0]

                # 1. Рисуем стандартную сетку
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )

                # 2. Выделяем и соединяем точки
                landmarks = face_landmarks.landmark
                def get_coords(idx):
                    lm = landmarks[idx]
                    return int(lm.x * image_width), int(lm.y * image_height)

                connection_color = (0, 255, 0) # Зеленый
                point_color = (0, 0, 255)      # Красный
                connection_thickness = 2
                point_radius = 4

                try:
                    # Соединяем глаза
                    cv2.line(annotated_image, get_coords(LEFT_EYE_INNER_CORNER), get_coords(LEFT_EYE_OUTER_CORNER), connection_color, connection_thickness)
                    cv2.line(annotated_image, get_coords(RIGHT_EYE_INNER_CORNER), get_coords(RIGHT_EYE_OUTER_CORNER), connection_color, connection_thickness)
                    # Соединяем челюсть
                    left_jaw_pt = get_coords(LEFT_JAW_ANGLE)
                    right_jaw_pt = get_coords(RIGHT_JAW_ANGLE)
                    cv2.line(annotated_image, left_jaw_pt, right_jaw_pt, connection_color, connection_thickness)
                    cv2.circle(annotated_image, left_jaw_pt, point_radius, point_color, -1) # Отметим точки
                    cv2.circle(annotated_image, right_jaw_pt, point_radius, point_color, -1)
                    # Рисуем точки
                    cv2.circle(annotated_image, get_coords(LEFT_TEMPLE), point_radius, point_color, -1)
                    cv2.circle(annotated_image, get_coords(RIGHT_TEMPLE), point_radius, point_color, -1)
                    cv2.circle(annotated_image, get_coords(NOSE_BRIDGE), point_radius, point_color, -1)
                    cv2.circle(annotated_image, get_coords(NOSE_BASE), point_radius, point_color, -1)
                except IndexError:
                     status_message = "Ошибка: Неверный индекс точки. Обновите MediaPipe?"
                except Exception as e:
                     status_message = f"Ошибка рисования: {e}"


        # Кодируем результат обратно в байты (JPEG)
        is_success, buffer = cv2.imencode(".jpg", annotated_image)
        if not is_success:
            return None, "Ошибка: Не удалось закодировать результат."

        processed_image_bytes = buffer.tobytes()
        return processed_image_bytes, status_message

    except Exception as e:
        # Ловим общие ошибки обработки
        print(f"Критическая ошибка в process_face_mesh: {e}")
        return None, f"Внутренняя ошибка сервера: {e}"


# --- Маршруты Flask ---

@app.route('/', methods=['GET'])
def index():
    """ Отображает главную страницу (форму загрузки) """
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
        # Читаем байты файла
        original_image_bytes = file.read()

        # Обрабатываем изображение
        processed_image_bytes, status_message = process_face_mesh(original_image_bytes)

        # Кодируем оба изображения в Base64 для отображения в HTML
        original_image_b64 = base64.b64encode(original_image_bytes).decode('utf-8')

        processed_image_b64 = None
        if processed_image_bytes:
             processed_image_b64 = base64.b64encode(processed_image_bytes).decode('utf-8')

        # Возвращаем страницу с результатами
        return render_template('index.html',
                               status_message=status_message,
                               original_image_data=original_image_b64,
                               processed_image_data=processed_image_b64)

    return render_template('index.html', error_message="Ошибка при загрузке файла.")


# --- Запуск приложения ---
# Обычно это делается через Gunicorn на сервере, но для локального теста можно так:
if __name__ == '__main__':
    # Используем порт 8080, как часто принято в облачных развертываниях
    # host='0.0.0.0' делает сервер доступным извне (важно для Docker/Render)
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=False)
    # Ставим debug=False для продакшена
