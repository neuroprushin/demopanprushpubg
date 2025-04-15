import os
import cv2
import mediapipe as mp
import numpy as np
import base64
import io
import time
import math
from flask import Flask, render_template, request

# --- Инициализация Flask ---
app = Flask(__name__)

# --- Настройка MediaPipe FaceLandmarker ---
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# --- Константы ---
MODEL_PATH = 'face_landmarker.task'
LEFT_TEMPLE = 139
RIGHT_TEMPLE = 368
LEFT_JAW = 58
RIGHT_JAW = 288
POINTS_TO_DRAW = [LEFT_TEMPLE, RIGHT_TEMPLE, LEFT_JAW, RIGHT_JAW]
# !!! ИЗМЕНЕНО: Уменьшаем порог для средней зоны !!!
TOLERANCE_PERCENTAGE = 0.015 # 1.5%

# --- Глобальная инициализация модели (без изменений) ---
face_landmarker_options = None
landmarker = None
try:
    if os.path.exists(MODEL_PATH):
         face_landmarker_options = FaceLandmarkerOptions(
             base_options=BaseOptions(model_asset_path=MODEL_PATH),
             running_mode=VisionRunningMode.IMAGE,
             num_faces=1,
             output_face_blendshapes=False,
             output_facial_transformation_matrixes=True) # Оставляем True для расчета угла
         landmarker = FaceLandmarker.create_from_options(face_landmarker_options)
         print(f"FaceLandmarker модель '{MODEL_PATH}' успешно загружена.")
    else:
         print(f"ОШИБКА: Файл модели FaceLandmarker не найден по пути: {MODEL_PATH}")
except Exception as e:
    print(f"ОШИБКА при инициализации FaceLandmarker: {e}")
    face_landmarker_options = None
    landmarker = None

# --- Вспомогательные функции (get_roll_angle, get_projected_x, draw_on_image - без изменений) ---
def get_roll_angle(rotation_matrix):
    """Вычисляет угол крена (roll) из матрицы вращения 3x3."""
    try:
        if rotation_matrix.shape == (3, 3):
             # Используем atan2(R[2,1], R[2,2]) для вращения вокруг оси взгляда (X)
             # Или atan2(R[1,0], R[0,0]) для вращения вокруг оси Z камеры (Yaw)
             # Нам нужен наклон влево-вправо, который обычно roll вокруг оси Z модели/мира
             # Попробуем снова atan2(R[1,0], R[0,0]) - это часто Yaw, но может быть тем, что нужно визуально
             # roll_angle_rad = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0]) # Yaw?
             # Попробуем roll = atan2( R[2,1], R[2,2] ) - вращение вокруг оси X мира?
             roll_angle_rad = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])

             # Конвертируем в градусы для лога
             roll_deg = math.degrees(roll_angle_rad)
             print(f"Roll angle radians: {roll_angle_rad:.4f}, degrees: {roll_deg:.2f}")
             # Добавим ограничение на угол, т.к. atan2 возвращает от -pi до pi
             # А нам нужен наклон от вертикали (0 = вертикально, >0 = вправо, <0 = влево)
             # Это похоже на roll_angle_rad как есть.
             return roll_angle_rad, roll_deg
        else:
             print("Ошибка: Матрица вращения не 3x3")
             return 0.0, 0.0
    except Exception as e:
        print(f"Ошибка вычисления угла: {e}")
        return 0.0, 0.0

def get_projected_x(xt, yt, y_target, theta):
    """Находит X координату на линии, проходящей через (xt, yt) под углом theta, на уровне y_target."""
    cos_theta = math.cos(theta)
    if abs(cos_theta) < 1e-6: return xt # Вертикальная линия
    tan_theta = math.tan(theta)
    if abs(tan_theta) < 1e-9: # Горизонтальная линия - проекция не имеет смысла или бесконечна? Вернем xt как заглушку.
         print("Предупреждение: Почти горизонтальная линия в get_projected_x")
         return xt
    projected_x = xt + (y_target - yt) / tan_theta
    return projected_x

def draw_on_image(image_np, detection_result, key_points_px):
    """Рисует сетку и ключевые точки на изображении NumPy."""
    annotated_image = image_np.copy()
    if detection_result and detection_result.face_landmarks:
        face_landmarks_list = detection_result.face_landmarks[0]
        try:
            from mediapipe.framework.formats import landmark_pb2
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks_list
            ])
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
        except Exception as e: print(f"Ошибка при рисовании сетки: {e}")
        for idx in POINTS_TO_DRAW:
            if idx in key_points_px: cv2.circle(annotated_image, key_points_px[idx], radius=4, color=(0, 0, 255), thickness=-1)
            else: print(f"Предупреждение: Нет пиксельных координат для точки {idx}")
    return annotated_image

# --- Основная функция обработки ---
def process_image_and_classify(image_bytes):
    """
    Главная функция: декодирует, запускает FaceLandmarker, классифицирует по пикселям, рисует.
    Возвращает: байты изображения, результат (строка), детали расчета (строка).
    """
    start_time = time.time()
    classification_result = "Не удалось обработать."
    debug_info = "" # Строка для деталей расчета
    processed_image_bytes = None

    if landmarker is None:
        return None, "Ошибка сервера: Модель не загружена.", "Модель FaceLandmarker не инициализирована."

    image_np = None
    mp_image = None
    detection_result = None
    annotated_image = None
    key_points_px = {}

    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image_np is None: return None, "Ошибка: Не удалось декодировать изображение.", ""

        image_height, image_width, _ = image_np.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
        detection_result = landmarker.detect(mp_image)

        if not detection_result or not detection_result.face_landmarks or not detection_result.facial_transformation_matrixes:
             classification_result = "Лицо не найдено или поза не определена."
             debug_info = "Не найдены ориентиры лица или матрица позы."
        else:
             face_landmarks_list = detection_result.face_landmarks[0]
             valid_points = True
             for idx in POINTS_TO_DRAW:
                 if idx < len(face_landmarks_list):
                     lm = face_landmarks_list[idx]
                     key_points_px[idx] = (int(lm.x * image_width), int(lm.y * image_height))
                 else:
                     valid_points = False
                     err_msg = f"Ошибка: Индекс точки {idx} вне диапазона."
                     classification_result = err_msg
                     debug_info = err_msg
                     break

             if valid_points:
                 pose_matrix_4x4 = detection_result.facial_transformation_matrixes[0]
                 rotation_matrix_3x3 = pose_matrix_4x4[:3, :3]
                 roll_angle_rad, roll_angle_deg = get_roll_angle(rotation_matrix_3x3)

                 (xtL, ytL) = key_points_px[LEFT_TEMPLE]
                 (xtR, ytR) = key_points_px[RIGHT_TEMPLE]
                 (xjL, yjL) = key_points_px[LEFT_JAW]
                 (xjR, yjR) = key_points_px[RIGHT_JAW]

                 pixel_inter_temple_width = math.sqrt((xtR - xtL)**2 + (ytR - ytL)**2)
                 if pixel_inter_temple_width < 1:
                     classification_result = "Ошибка: Слишком малое расстояние между висками."
                     debug_info = f"Расстояние между висками: {pixel_inter_temple_width:.1f}px"
                 else:
                     pixel_tolerance = TOLERANCE_PERCENTAGE * pixel_inter_temple_width
                     # Угол линии = pi/2 - roll_angle_rad (от оси +X)
                     line_angle_rad = math.pi / 2.0 - roll_angle_rad

                     x_line_L = get_projected_x(xtL, ytL, yjL, line_angle_rad)
                     x_line_R = get_projected_x(xtR, ytR, yjR, line_angle_rad)

                     pixel_deviation_L = xjL - x_line_L # < 0 => шире
                     pixel_deviation_R = xjR - x_line_R # > 0 => шире

                     if pixel_deviation_L < -pixel_tolerance: class_L = "Широкая"
                     elif pixel_deviation_L > pixel_tolerance: class_L = "Узкая"
                     else: class_L = "Средняя"

                     if pixel_deviation_R > pixel_tolerance: class_R = "Широкая"
                     elif pixel_deviation_R < -pixel_tolerance: class_R = "Узкая"
                     else: class_R = "Средняя"

                     if class_L == "Широкая" and class_R == "Широкая": classification_result = "Широкая"
                     elif class_L == "Узкая" and class_R == "Узкая": classification_result = "Узкая"
                     else: classification_result = "Средняя"

                     # --- Формируем строку с деталями расчета ---
                     debug_info = (
                         f"Параметры расчета (пиксели):\n"
                         f"  Расстояние между висками: {pixel_inter_temple_width:.1f} px\n"
                         f"  Порог толерантности ({TOLERANCE_PERCENTAGE*100:.1f}%): {pixel_tolerance:.1f} px\n"
                         f"  Угол наклона головы (Roll): {roll_angle_deg:.1f} deg\n"
                         f"  Угол линий от вертикали: {math.degrees(line_angle_rad - math.pi/2):.1f} deg\n"
                         f"Левая сторона:\n"
                         f"  X виска: {xtL}, Y виска: {ytL}\n"
                         f"  X челюсти: {xjL}, Y челюсти: {yjL}\n"
                         f"  X на линии @ Y челюсти: {x_line_L:.1f}\n"
                         f"  Отклонение X: {pixel_deviation_L:.1f} px ({class_L})\n"
                         f"Правая сторона:\n"
                         f"  X виска: {xtR}, Y виска: {ytR}\n"
                         f"  X челюсти: {xjR}, Y челюсти: {yjR}\n"
                         f"  X на линии @ Y челюсти: {x_line_R:.1f}\n"
                         f"  Отклонение X: {pixel_deviation_R:.1f} px ({class_R})\n"
                         f"ИТОГ: {classification_result}"
                     )
                     print(debug_info) # Печатаем также в консоль сервера

        # --- Рисование ---
        annotated_image = draw_on_image(image_np, detection_result, key_points_px)
        if classification_result.startswith("Лицо не найдено"):
             cv2.putText(annotated_image, "Face/Pose not detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Кодируем результат в JPEG байты
        is_success, buffer = cv2.imencode(".jpg", annotated_image)
        if is_success:
            processed_image_bytes = buffer.tobytes()
        else:
            classification_result = "Ошибка: Не удалось закодировать результат."
            debug_info = "Не удалось закодировать итоговое изображение."
            processed_image_bytes = None

    except Exception as e:
        print(f"Критическая ошибка в process_image_and_classify: {e}")
        import traceback
        traceback.print_exc()
        if classification_result == "Не удалось обработать.": classification_result = f"Внутренняя ошибка: {e}"
        debug_info = f"Критическая ошибка: {e}\n{traceback.format_exc()}" # Добавляем трейсбек в дебаг
        processed_image_bytes = None

    finally:
        # Очистка памяти
        del image_np, mp_image, detection_result, annotated_image, key_points_px
        end_time = time.time()
        print(f"Время обработки: {end_time - start_time:.4f} сек.")

    # Возвращаем байты, статус и строку с дебаг-информацией
    return processed_image_bytes, classification_result, debug_info


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
        # Получаем три значения из функции
        processed_image_bytes, status_message, debug_info = process_image_and_classify(original_image_bytes)
        del original_image_bytes

        processed_image_b64 = None
        if processed_image_bytes:
             processed_image_b64 = base64.b64encode(processed_image_bytes).decode('utf-8')
             del processed_image_bytes

        # Передаем все три значения в шаблон
        return render_template('index.html',
                               status_message=status_message,
                               processed_image_data=processed_image_b64,
                               debug_info=debug_info) # Добавили debug_info

    return render_template('index.html', error_message="Ошибка при загрузке файла.")

# --- Запуск приложения (без изменений) ---
if __name__ == '__main__':
    if landmarker:
        print("Запуск Flask сервера...")
        app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=False)
    else:
        print("ОШИБКА: Не удалось загрузить модель FaceLandmarker. Сервер не запущен.")
