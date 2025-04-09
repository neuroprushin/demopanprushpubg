import os
import cv2
import numpy as np
import base64
from flask import Flask, render_template, request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

app = Flask(__name__)

# Настройка FaceLandmarker
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

# Индексы точек
LEFT_TEMPLE = 139
RIGHT_TEMPLE = 368
LEFT_JAW = 58
RIGHT_JAW = 288
POINTS_TO_DRAW = [LEFT_TEMPLE, RIGHT_TEMPLE, LEFT_JAW, RIGHT_JAW]

TOLERANCE_PERCENTAGE = 0.03

def process_image_and_classify(image_bytes):
    classification_result = "Ошибка: Не удалось обработать."
    processed_image_bytes = None

    try:
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return None, "Ошибка: Не удалось декодировать изображение."

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = face_landmarker.detect(mp_image)

        if not result.face_landmarks:
            classification_result = "Лицо не найдено."
            cv2.putText(image, "Face not detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            is_success, buffer = cv2.imencode(".jpg", image)
            processed_image_bytes = buffer.tobytes() if is_success else None
            return processed_image_bytes, classification_result

        face_landmarks = result.face_landmarks[0]
        height, width, _ = image.shape

        # 3D-координаты из FaceLandmarker (в относительных единицах)
        landmarks_3d = []
        for lm in face_landmarks:
            x = lm.x * width
            y = lm.y * height
            z = lm.z * width  # Z тоже в относительных единицах
            landmarks_3d.append([x, y, z])

        # Классификация
        X_temple_L = landmarks_3d[LEFT_TEMPLE][0]
        X_temple_R = landmarks_3d[RIGHT_TEMPLE][0]
        X_jaw_L = landmarks_3d[LEFT_JAW][0]
        X_jaw_R = landmarks_3d[RIGHT_JAW][0]

        inter_temple_width = abs(X_temple_R - X_temple_L)
        if inter_temple_width < 1e-6:
            classification_result = "Ошибка: Нулевая ширина висков."
        else:
            tolerance = TOLERANCE_PERCENTAGE * inter_temple_width

            deviation_L = X_jaw_L - X_temple_L
            deviation_R = X_jaw_R - X_temple_R

            if deviation_L < -tolerance:
                class_L = "Широкая"
            elif deviation_L > tolerance:
                class_L = "Узкая"
            else:
                class_L = "Средняя"

            if deviation_R > tolerance:
                class_R = "Широкая"
            elif deviation_R < -tolerance:
                class_R = "Узкая"
            else:
                class_R = "Средняя"

            if class_L == "Широкая" and class_R == "Широкая":
                classification_result = "Широкая"
            elif class_L == "Узкая" and class_R == "Узкая":
                classification_result = "Узкая"
            else:
                classification_result = "Средняя"

        # Рисование сетки и точек
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_face_mesh = mp.solutions.face_mesh
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )

        for idx in POINTS_TO_DRAW:
            lm = face_landmarks[idx]
            x_px = int(lm.x * width)
            y_px = int(lm.y * height)
            cv2.circle(image, (x_px, y_px), 3, (0, 0, 255), -1)

        is_success, buffer = cv2.imencode(".jpg", image)
        processed_image_bytes = buffer.tobytes() if is_success else None

    except Exception as e:
        print(f"Ошибка: {e}")
        classification_result = f"Внутренняя ошибка: {e}"

    return processed_image_bytes, classification_result

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

        processed_image_b64 = None
        if processed_image_bytes:
            processed_image_b64 = base64.b64encode(processed_image_bytes).decode('utf-8')

        return render_template('index.html', status_message=status_message, processed_image_data=processed_image_b64)

    return render_template('index.html', error_message="Ошибка при загрузке файла.")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=False)
