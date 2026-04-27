import cv2
import mediapipe as mp
import numpy as np
import copy
import os

# 1. CONFIGURACIÓN
# ----------------
DATA_PATH = 'data_dinamico'  # Carpeta raíz para los datos dinámicos
SEQUENCE_LENGTH = 30         # Número de frames por gesto
min_detection_confidence = 0.7
min_tracking_confidence = 0.5

# Mapeo de teclas a Letras (Según tu petición)
# Teclas 1-5 corresponden a los índices 0-4
ACTIONS = np.array(['CERVEZA', 'OLIVAS', 'CEBOLLAS', 'MADURO', 'LEBRON'])

# Mapa de teclas ASCII para cv2.waitKey()
# '1': 49, '2': 50, '3': 51, '4': 52, '5': 53
KEY_MAP = {
    49: 0, # Tecla 1 -> CERVEZA
    50: 1, # Tecla 2 -> OLIVAS
    51: 2, # Tecla 3 -> CEBOLLAS
    52: 3, # Tecla 4 -> MADURO
    53: 4  # Tecla 5 -> LEBRON
}

# Crear carpetas si no existen
for action in ACTIONS:
    action_path = os.path.join(DATA_PATH, action)
    if not os.path.exists(action_path):
        os.makedirs(action_path)

# 2. INICIALIZAR MEDIAPIPE
# ------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=min_detection_confidence,
    min_tracking_confidence=min_tracking_confidence,
)
mp_drawing = mp.solutions.drawing_utils

# 3. FUNCIONES DE NORMALIZACIÓN (Misma lógica que tu script anterior)
# -----------------------------
def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convertir a relativas
    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]
    for index, landmark_point in enumerate(temp_landmark_list):
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Aplanar
    flatten_list = []
    for temp_point in temp_landmark_list:
        flatten_list.append(temp_point[0])
        flatten_list.append(temp_point[1])

    # Normalización Absoluta
    max_value = max(list(map(abs, flatten_list)))
    def normalize_(n):
        return n / max_value if max_value != 0 else 0

    flatten_list = list(map(normalize_, flatten_list))
    return flatten_list

# 4. BUCLE PRINCIPAL
# ------------------
cap = cv2.VideoCapture(0)

print(f">>> SISTEMA LISTO PARA SECUENCIAS ({SEQUENCE_LENGTH} frames).")
print(f">>> Teclas: 1=CERVEZA, 2=OLIVOS, 3=CEBOLLAS, 4=MADURO, 5=LEBRON")
print(">>> Pulsa ESC para salir.")

# Variables de estado para la grabación
is_recording = False
action_num = -1     # Índice de la letra actual (0 a 4)
sequence_buffer = [] # Lista temporal para guardar los frames
frame_count = 0     # Contador de frames grabados en la secuencia actual

while True:
    ret, image = cap.read()
    if not ret:
        break

    image = cv2.flip(image, 1)
    debug_image = copy.deepcopy(image)

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image.flags.writeable = True

    current_frame_landmarks = [] # Landmarks procesados de ESTE frame

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmark_list = []
            for _, landmark in enumerate(hand_landmarks.landmark):
                landmark_list.append([landmark.x, landmark.y])

            # Normalizamos igual que en tu modelo estático
            pre_processed_landmark_list = pre_process_landmark(landmark_list)
            current_frame_landmarks = pre_processed_landmark_list

            mp_drawing.draw_landmarks(
                debug_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # --- LÓGICA DE GRABACIÓN DE SECUENCIA ---
    if is_recording:
        # Solo grabamos si detectamos mano. Si se pierde la mano, cancelamos para no tener "ruido"
        if len(current_frame_landmarks) > 0:
            sequence_buffer.append(current_frame_landmarks)
            frame_count += 1
        else:
            print("!!! Mano perdida durante la grabación. Cancelando secuencia.")
            is_recording = False
            sequence_buffer = []
            frame_count = 0

        # Verificar si hemos llegado a 30 frames
        if frame_count == SEQUENCE_LENGTH:
            # Construir ruta de guardado
            action_name = ACTIONS[action_num]
            folder_path = os.path.join(DATA_PATH, action_name)

            # Contar cuántos archivos hay ya para no sobrescribir
            # Formato nombre: "letra_numero.npy"
            existing_files = len(os.listdir(folder_path))
            npy_path = os.path.join(folder_path, f"{action_name}_{existing_files}.npy")

            # Guardar array numpy
            np_data = np.array(sequence_buffer) # Shape será (30, 42)
            np.save(npy_path, np_data)

            print(f"*** GUARDADA SECUENCIA: {action_name} -> {npy_path}")

            # Resetear
            is_recording = False
            sequence_buffer = []
            frame_count = 0

    # --- INTERFAZ VISUAL ---
    # Barra de estado superior
    cv2.rectangle(debug_image, (0,0), (640, 40), (0,0,0), -1)

    if is_recording:
        msg = f"GRABANDO {ACTIONS[action_num]}: {frame_count}/{SEQUENCE_LENGTH}"
        color = (0, 0, 255) # Rojo
    else:
        msg = "ESPERANDO TECLA (1-5)..."
        color = (200, 200, 200) # Gris

    cv2.putText(debug_image, msg, (10, 28),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    cv2.imshow('Proyecto chorra Dynamic Collector', debug_image)

    # --- CAPTURA DE TECLAS ---
    key = cv2.waitKey(10)

    if key == 27: # ESC
        break

    # Si NO estamos grabando, verificamos si se pulsa 1-5
    if not is_recording and key in KEY_MAP:
        action_num = KEY_MAP[key]
        is_recording = True
        sequence_buffer = []
        frame_count = 0
        print(f"--- Iniciando grabación para: {ACTIONS[action_num]} ---")

cap.release()
cv2.destroyAllWindows()
