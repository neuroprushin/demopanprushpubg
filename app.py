import os
import cv2
import mediapipe as mp
import numpy as np
import base64
import io
import time
import math # Нужен для тригонометрии
from flask import Flask, render_template, request

# --- Инициализация Flask ---
app = Flask(__name__)

# --- Настройка MediaPipe FaceLandmarker ---
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
# Для удобства доступа к утилитам рисования
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh # Нужно для FACEMESH_TESSELATION

# --- Константы ---
# Используем базовую модель без blendshapes
MODEL_PATH = 'face_landmarker.task'
# Индексы ключевых точек
LEFT_TEMPLE = 139
RIGHT_TEMPLE = 368
LEFT_JAW = 58
RIGHT_JAW = 288
POINTS_TO_DRAW = [LEFT_TEMPLE, RIGHT_TEMPLE, LEFT_JAW, RIGHT_JAW]
# Порог для классификации (% от пиксельной ширины висков)
TOLERANCE_PERCENTAGE = 0.03 # 3%

# --- Глобальная инициализация модели ---
face_landmarker_options = None
landmarker = None
try:
    if os.path.exists(MODEL_PATH):
         face_landmarker_options = FaceLandmarkerOptions(
             base_options=BaseOptions(model_asset_path=MODEL_PATH),
             running_mode=VisionRunningMode.IMAGE,
             num_faces=1,
             output_face_blendshapes=False,
             # !!! Включаем вывод матрицы позы !!!
             output_facial_transformation_matrixes=True)
         landmarker = FaceLandmarker.create_from_options(face_landmarker_options)
         print(f"FaceLandmarker модель '{MODEL_PATH}' успешно загружена.")
    else:
         print(f"ОШИБКА: Файл модели FaceLandmarker не найден по пути: {MODEL_PATH}")
except Exception as e:
    print(f"ОШИБКА при инициализации FaceLandmarker: {e}")
    face_landmarker_options = None
    landmarker = None

# --- Вспомогательные функции ---

def get_roll_angle(rotation_matrix):
    """Вычисляет угол крена (roll) из матрицы вращения 3x3."""
    # Формула для roll (вращение вокруг оси Z, если смотреть из камеры)
    # sin(roll) = R[1][0], cos(roll) = R[0][0]
    # Или можно использовать atan2(R[1][0], R[0][0])
    # Но часто в компьютерном зрении roll - это вращение вокруг оси, направленной ВДОЛЬ взгляда (ось Z модели)
    # Для этого используются другие элементы матрицы: atan2(R[2][1], R[2][2]) ??? Зависит от конвенции.
    # Попробуем самый частый вариант для "наклона головы влево-вправо" при взгляде на экран:
    # Это вращение вокруг оси Y (pitch) или X (roll)?
    # Наклон влево-вправо - это обычно roll. В стандартной Tait-Bryan ZYX конвенции:
    # sy = sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
    # roll = atan2(R[2,1], R[2,2])
    # pitch = atan2(-R[2,0], sy)
    # yaw = atan2(R[1,0], R[0,0])
    # Нам нужен roll.
    try:
        # Убедимся что матрица 3x3
        if rotation_matrix.shape == (3, 3):
             # Roll (вращение вокруг оси Z в локальных координатах, ~наклон влево-вправо)
             roll = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
             # Или Roll (вращение вокруг оси X в локальных координатах, ~наклон вперед-назад)?
             # roll_x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
             # Давай использовать roll = atan2(R[1,0], R[0,0]), который часто соответствует вращению "плоскости лица"
             # Но возможно, нам нужен угол, перпендикулярный взгляду камеры.
             # Проверим R[2,1] vs R[2,2] - это вращение вокруг оси Z камеры?
             roll_cam_z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0]) # Yaw камеры?
             roll_face = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2]) # Вращение вокруг Z лица?

             # Используем roll_face, который должен отвечать за наклон лица в его плоскости
             # Конвертируем в градусы для лога
             print(f"Roll angle (Z face) radians: {roll_face:.4f}, degrees: {math.degrees(roll_face):.2f}")
             return roll_face
        else:
             print("Ошибка: Матрица вращения не 3x3")
             return 0.0 # Возвращаем 0, если не можем вычислить
    except Exception as e:
        print(f"Ошибка вычисления угла: {e}")
        return 0.0


def get_projected_x(xt, yt, y_target, theta):
    """Находит X координату на линии, проходящей через (xt, yt) под углом theta, на уровне y_target."""
    # theta - угол в радианах относительно положительной оси X
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    # Проверяем на почти вертикальную линию (cos(theta) близок к 0)
    if abs(cos_theta) < 1e-6:
        # Линия вертикальна, X совпадает с xt
        return xt
    else:
        # Уравнение линии: (y - yt) = tan(theta) * (x - xt)
        # x = xt + (y - yt) / tan(theta) = xt + (y - yt) * cos(theta) / sin(theta)
        # Проверим на почти горизонтальную линию (sin(theta) близок к 0), но это не должно вызвать деления на 0 тут
        tan_theta = math.tan(theta) # Используем tan, т.к. уже проверили на вертикальность
        projected_x = xt + (y_target - yt) / tan_theta
        return projected_x

def draw_on_image(image_np, detection_result, key_points_px):
    """Рисует сетку и ключевые точки на изображении NumPy."""
    annotated_image = image_np.copy()
    # image_height, image_width, _ = annotated_image.shape # Не используем здесь

    if detection_result and detection_result.face_landmarks:
        face_landmarks_list = detection_result.face_landmarks[0]

        # --- Рисуем сетку ---
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
        except Exception as e:
            print(f"Ошибка при рисовании сетки: {e}")

        # --- Рисуем ключевые точки (используем переданные пиксельные координаты) ---
        for idx in POINTS_TO_DRAW:
            if idx in key_points_px:
                cv2.circle(annotated_image, key_points_px[idx], radius=4, color=(0, 0, 255), thickness=-1) # Красные точки крупнее
            else:
                 print(f"Предупреждение: Нет пиксельных координат для точки {idx}")

    return annotated_image


# --- Основная функция обработки ---
def process_image_and_classify(image_bytes):
    """
    Главная функция: декодирует, запускает FaceLandmarker, классифицирует по пикселям, рисует.
    Возвращает байты обработанного изображения и строку с результатом/ошибкой.
    """
    start_time = time.time()
    classification_result = "Не удалось обработать."
    processed_image_bytes = None

    if landmarker is None:
        print("ОШИБКА: FaceLandmarker не инициализирован.")
        return None, "Ошибка сервера: Модель не загружена."

    image_np = None
    mp_image = None
    detection_result = None
    annotated_image = None
    key_points_px = {} # Словарь для хранения пиксельных координат

    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image_np is None: return None, "Ошибка: Не удалось декодировать изображение."

        image_height, image_width, _ = image_np.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
        detection_result = landmarker.detect(mp_image)

        # --- Классификация и подготовка данных для рисования ---
        if not detection_result or not detection_result.face_landmarks or not detection_result.facial_transformation_matrixes:
             classification_result = "Лицо не найдено или поза не определена."
        else:
             face_landmarks_list = detection_result.face_landmarks[0]
             # Денормализуем координаты нужных точек
             valid_points = True
             for idx in POINTS_TO_DRAW:
                 if idx < len(face_landmarks_list):
                     lm = face_landmarks_list[idx]
                     key_points_px[idx] = (int(lm.x * image_width), int(lm.y * image_height))
                 else:
                     print(f"Ошибка: Индекс точки {idx} вне диапазона.")
                     valid_points = False
                     classification_result = f"Ошибка: Не найдена точка {idx}."
                     break # Прерываем, если не все точки найдены

             if valid_points:
                 # Получаем матрицу позы и вычисляем угол крена
                 pose_matrix_4x4 = detection_result.facial_transformation_matrixes[0]
                 rotation_matrix_3x3 = pose_matrix_4x4[:3, :3]
                 roll_angle_rad = get_roll_angle(rotation_matrix_3x3) # Угол наклона лица

                 # Пиксельные координаты
                 (xtL, ytL) = key_points_px[LEFT_TEMPLE]
                 (xtR, ytR) = key_points_px[RIGHT_TEMPLE]
                 (xjL, yjL) = key_points_px[LEFT_JAW]
                 (xjR, yjR) = key_points_px[RIGHT_JAW]

                 # Считаем ширину висков в пикселях и порог
                 pixel_inter_temple_width = math.sqrt((xtR - xtL)**2 + (ytR - ytL)**2)
                 if pixel_inter_temple_width < 1: # Защита от слишком малых лиц
                     classification_result = "Ошибка: Слишком малое расстояние между висками."
                 else:
                     pixel_tolerance = TOLERANCE_PERCENTAGE * pixel_inter_temple_width

                     # Угол для "вертикальных" линий с учетом наклона
                     # Угол 0 - вправо, pi/2 - вверх. Нам нужно вниз (3pi/2 или -pi/2) + наклон roll_angle_rad
                     # Вертикаль вниз = -pi/2. С учетом крена theta = -pi/2 + roll_angle_rad ?
                     # Или проще: Вертикальная линия в системе лица повернута на roll_angle_rad
                     # Угол линии = pi/2 + roll_angle_rad (если roll=0, угол=pi/2, т.е. вертикально) ???
                     # Проверим: если roll > 0 (наклон вправо), линия должна наклониться вправо.
                     # Угол theta измеряется от оси X. Вертикаль - это pi/2.
                     # Если голова наклонена вправо (roll > 0), линия тоже должна наклониться вправо, т.е. угол theta должен УМЕНЬШИТЬСЯ от pi/2.
                     # Если голова наклонена влево (roll < 0), линия наклоняется влево, theta УВЕЛИЧИВАЕТСЯ от pi/2.
                     # Значит, theta = pi/2 - roll_angle_rad
                     line_angle_rad = math.pi / 2.0 - roll_angle_rad

                     # Считаем проекции X на линии на уровне Y челюсти
                     x_line_L = get_projected_x(xtL, ytL, yjL, line_angle_rad)
                     x_line_R = get_projected_x(xtR, ytR, yjR, line_angle_rad)

                     # Считаем пиксельные отклонения
                     pixel_deviation_L = xjL - x_line_L # < 0 значит челюсть левее/шире
                     pixel_deviation_R = xjR - x_line_R # > 0 значит челюсть правее/шире

                     # Классифицируем стороны
                     if pixel_deviation_L < -pixel_tolerance: class_L = "Широкая"
                     elif pixel_deviation_L > pixel_tolerance: class_L = "Узкая"
                     else: class_L = "Средняя"

                     if pixel_deviation_R > pixel_tolerance: class_R = "Широкая"
                     elif pixel_deviation_R < -pixel_tolerance: class_R = "Узкая"
                     else: class_R = "Средняя"

                     # Итоговая классификация
                     if class_L == "Широкая" and class_R == "Широкая": classification_result = "Широкая"
                     elif class_L == "Узкая" and class_R == "Узкая": classification_result = "Узкая"
                     else: classification_result = "Средняя"

                     print(f"Классификация (Пиксели): Шир.виск={pixel_inter_temple_width:.1f}px, Порг={pixel_tolerance:.1f}px, Откл.L={pixel_deviation_L:.1f}px ({class_L}), Откл.R={pixel_deviation_R:.1f}px ({class_R}) -> ИТОГ={classification_result}")


        # --- Рисование ---
        # Рисуем всегда, передавая image_np и найденные точки/результат
        annotated_image = draw_on_image(image_np, detection_result, key_points_px)
        if classification_result == "Лицо не найдено или поза не определена.":
             cv2.putText(annotated_image, "Face/Pose not detected", (50, 50),
                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Кодируем результат в JPEG байты
        is_success, buffer = cv2.imencode(".jpg", annotated_image)
        if is_success:
            processed_image_bytes = buffer.tobytes()
        else:
            classification_result = "Ошибка: Не удалось закодировать результат."
            processed_image_bytes = None

    except Exception as e:
        print(f"Ошибка в process_image_and_classify: {e}")
        import traceback
        traceback.print_exc() # Печатаем полный трейсбек для диагностики
        if classification_result == "Не удалось обработать.":
             classification_result = f"Внутренняя ошибка: {e}"
        processed_image_bytes = None

    finally:
        # Очистка памяти
        del image_np
        del mp_image
        del detection_result
        del annotated_image
        del key_points_px # Очищаем словарь с координатами
        end_time = time.time()
        print(f"Время обработки: {end_time - start_time:.4f} сек.")

    return processed_image_bytes, classification_result

# --- Маршруты Flask (без изменений) ---
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
        # Освобождаем память от исходных байт
        del original_image_bytes

        processed_image_b64 = None
        if processed_image_bytes:
             processed_image_b64 = base64.b64encode(processed_image_bytes).decode('utf-8')
             # Освобождаем и эти байты после кодирования
             del processed_image_bytes

        return render_template('index.html',
                               status_message=status_message,
                               processed_image_data=processed_image_b64)

    return render_template('index.html', error_message="Ошибка при загрузке файла.")

# --- Запуск приложения (без изменений) ---
if __name__ == '__main__':
    if landmarker:
        print("Запуск Flask сервера...")
        app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=False)
    else:
        print("ОШИБКА: Не удалось загрузить модель FaceLandmarker. Сервер не запущен.")
