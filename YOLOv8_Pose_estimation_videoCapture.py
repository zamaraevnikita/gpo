import logging
import cv2
from ultralytics import YOLO
import numpy as np
import json
import torch

logging.getLogger('ultralytics').setLevel(logging.ERROR)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Функция для вычисления центра тяжести
def calculate_center(keypoints):
    if len(keypoints) == 0:
        return None
    mean_x = np.mean(keypoints[:, 0])
    mean_y = np.mean(keypoints[:, 1])
    return float(mean_x), float(mean_y)

def calculate_average_keypoints(keypoints_buffer):
    mean_keypoints = {}
    for key in keypoints_buffer[0].keys():
        points = np.array([kp[key] for kp in keypoints_buffer])
        mean_keypoints[key] = np.mean(points, axis=0)
    return mean_keypoints

def calculate_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

# Функция для вычисления разницы в длине ног
def leg_length_difference(keypoints):
    left_leg_length = np.linalg.norm(keypoints['left_ankle'] - keypoints['left_hip'])
    right_leg_length = np.linalg.norm(keypoints['right_ankle'] - keypoints['right_hip'])
    return float(abs(left_leg_length - right_leg_length))


# Высота плеч (разница в высоте между левым и правым плечом)
def shoulder_height_difference(keypoints):
    return float(abs(keypoints['left_shoulder'][1] - keypoints['right_shoulder'][1]))

# Наклон таза (разница в высоте между левым и правым бедром)
def pelvis_tilt(keypoints):
    return float(abs(keypoints['left_hip'][1] - keypoints['right_hip'][1]))

# Вальгус/варус коленей (угол отклонения коленей внутрь или наружу)
def knee_valgus_varus(keypoints):
    left_knee_angle = calculate_angle(keypoints['left_hip'], keypoints['left_knee'], keypoints['left_ankle'])
    right_knee_angle = calculate_angle(keypoints['right_hip'], keypoints['right_knee'], keypoints['right_ankle'])
    return float(left_knee_angle), float(right_knee_angle)

# Ширина шага (расстояние между стопами)
def step_width(keypoints):
    return float(abs(keypoints['left_ankle'][0] - keypoints['right_ankle'][0]))

# Асимметрия шага (разница в длине шага между ногами)
def step_asymmetry(keypoints):
    left_step_length = np.linalg.norm(keypoints['left_ankle'] - keypoints['left_knee'])
    right_step_length = np.linalg.norm(keypoints['right_ankle'] - keypoints['right_knee'])
    return float(abs(left_step_length - right_step_length))

# Ротация таза и плеч (угол ротации таза и плеч по отношению к центральной оси)
def pelvis_rotation(keypoints):
    pelvis_angle = calculate_angle(keypoints['left_hip'], keypoints['right_hip'], [keypoints['center_of_mass'][0], 0])
    return float(pelvis_angle)

def shoulder_rotation(keypoints):
    shoulder_angle = calculate_angle(keypoints['left_shoulder'], keypoints['right_shoulder'], [keypoints['center_of_mass'][0], 0])
    return float(shoulder_angle)

# Отклонение центра тяжести (от средней линии тела)
def center_of_gravity_deviation(keypoints):
    midpoint_hip = (keypoints['left_hip'] + keypoints['right_hip']) / 2  # Рассчитываем середину между бедрами
    return float(abs(keypoints['center_of_mass'][0] - midpoint_hip[0]))

# Симметрия движения рук (разница в амплитуде движения рук)
def arm_movement_symmetry(keypoints):
    left_arm_length = np.linalg.norm(keypoints['left_shoulder'] - keypoints['left_hand'])
    right_arm_length = np.linalg.norm(keypoints['right_shoulder'] - keypoints['right_hand'])
    return float(abs(left_arm_length - right_arm_length))

# Вспомогательная функция для расчета угла между тремя точками
def calculate_angle(point1, point2, point3):
    a = np.array(point1)
    b = np.array(point2)
    c = np.array(point3)
    ab = a - b
    cb = c - b
    cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

# Функция для обработки видео и сохранения аннотированного видео с расчетами в JSON
def process_video(video_path, output_video_path, yolo_model='yolo11n-pose.pt', confidence_threshold=0.5, avg_frame_interval=10):
    model = YOLO(yolo_model).to(device)

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    video_data = []
    averaged_steps_data = {}
    frame_number = 0
    keypoints_buffer = []
    step_count = 0  # Начнем с 0, чтобы правильно отслеживать шаги

    previous_left_ankle = None
    previous_right_ankle = None
    current_step_distance = 0
    min_distance_threshold = 30  # Уменьшенное значение порога дистанции
    min_frame_gap = 10  # Уменьшенный интервал между шагами
    last_step_frame = 0
    step_in_progress = False  # Отслеживаем, идет ли шаг

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=confidence_threshold)
        keypoints = results[0].keypoints.xy.cpu().numpy().reshape(-1, 2)

        keypoints_dict = {
            'left_ankle': keypoints[15],
            'right_ankle': keypoints[16],
            'left_knee': keypoints[13],
            'right_knee': keypoints[14],
            'left_hip': keypoints[11],
            'right_hip': keypoints[12],
            'left_shoulder': keypoints[5],
            'right_shoulder': keypoints[6],
            'left_hand': keypoints[9],
            'right_hand': keypoints[10],
            'center_of_mass': calculate_center(keypoints)
        }

        keypoints_buffer.append(keypoints_dict)

        if (frame_number + 1) % avg_frame_interval == 0:
            avg_keypoints = calculate_average_keypoints(keypoints_buffer)
            keypoints_buffer = []

            frame_data = {
                "leg_length_difference": leg_length_difference(avg_keypoints),
                "shoulder_height_difference": shoulder_height_difference(avg_keypoints),
                "pelvis_tilt": pelvis_tilt(avg_keypoints),
                "knee_valgus_varus": knee_valgus_varus(avg_keypoints),
                "step_width": step_width(avg_keypoints),
                "step_asymmetry": step_asymmetry(avg_keypoints),
                "pelvis_rotation": pelvis_rotation(avg_keypoints),
                "shoulder_rotation": shoulder_rotation(avg_keypoints),
                "center_of_gravity_deviation": center_of_gravity_deviation(avg_keypoints),
                "arm_movement_symmetry": arm_movement_symmetry(avg_keypoints),
                "step_count": step_count  # Здесь записываем количество шагов
            }

            frame_data_converted = {k: float(v) if isinstance(v, np.float32) else v for k, v in frame_data.items()}
            video_data.append(frame_data_converted)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)

        current_step_distance = calculate_distance(keypoints_dict['left_ankle'], keypoints_dict['right_ankle'])

        if previous_left_ankle is not None and previous_right_ankle is not None:
            if current_step_distance > min_distance_threshold and not step_in_progress:  # Проверяем, идет ли шаг
                if frame_number - last_step_frame > min_frame_gap:
                    step_count += 1
                    print(f"Шаг {step_count} зафиксирован на кадре {frame_number}")
                    averaged_steps_data[f"шаг {step_count}"] = frame_data_converted  # Сохраняем шаг в JSON
                    last_step_frame = frame_number
                    step_in_progress = True  # Устанавливаем флаг, что шаг зафиксирован

            # Если расстояние между лодыжками меньше порога, сбрасываем флаг
            elif current_step_distance < min_distance_threshold:
                step_in_progress = False

        previous_left_ankle = keypoints_dict['left_ankle']
        previous_right_ankle = keypoints_dict['right_ankle']
        frame_number += 1

    # Сохраняем данные в JSON
    with open('video_analysis.json', 'w') as json_file:
        json.dump(video_data, json_file, indent=4)

    with open('averaged_steps_data.json', 'w') as steps_json_file:
        json.dump(averaged_steps_data, steps_json_file, indent=4)

    cap.release()
    out.release()



if __name__ == "__main__":
    video_path = 'video.mp4'
    output_video_path = 'annotated_video.mp4'
    process_video(video_path, output_video_path, confidence_threshold=0.5)