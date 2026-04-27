import cv2
import mediapipe as mp
import csv
import copy
import os

# 1. CONFIGURACIÓN
# ----------------
dataset_file = 'data/gestos_dataset.csv'
min_detection_confidence = 0.7
min_tracking_confidence = 0.5

# Mapeo de letras (a=0, b=1, ..., z=26)
# Esto nos servirá para mostrar la letra en pantalla en lugar del número
ALPHABET = "abcdefghijklmnopqrstuvwxyz"

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

# 3. GESTIÓN DE CONTADORES
# ------------------------
def load_counts(csv_path):
    """Lee el CSV y cuenta cuántas muestras hay por cada clase."""
    counts = {i: 0 for i in range(len(ALPHABET))} # Inicializa todo a 0

    if not os.path.exists(csv_path):
        return counts

    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                try:
                    # Asumimos que la primera columna es el ID de la clase
                    label = int(row[0])
                    if label in counts:
                        counts[label] += 1
                except (ValueError, IndexError):
                    continue
    except Exception as e:
        print(f"Error leyendo contadores: {e}")

    return counts

# Cargamos los contadores al inicio
class_counts = load_counts(dataset_file)
print(f"Dataset cargado. Resumen actual: {class_counts}")

# 4. FUNCIONES DE NORMALIZACIÓN
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

def logging_csv(class_id, landmark_list):
    """ Guarda el ID y los datos en el CSV """
    if 0 <= class_id < len(ALPHABET):
        csv_path = dataset_file
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([class_id, *landmark_list])
        return True
    return False

# 5. BUCLE PRINCIPAL
# ------------------
cap = cv2.VideoCapture(0)

# Crear archivo si no existe
if not os.path.exists(dataset_file):
    with open(dataset_file, 'w', newline="") as f:
        pass

print(">>> SISTEMA LISTO.")
print(">>> Pulsa las teclas 'a'-'z' para guardar gestos.")
print(">>> Pulsa ESC para salir.")

last_saved_msg = "Esperando..."
last_saved_color = (200, 200, 200)

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

    current_landmarks = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmark_list = []
            for _, landmark in enumerate(hand_landmarks.landmark):
                landmark_list.append([landmark.x, landmark.y])

            pre_processed_landmark_list = pre_process_landmark(landmark_list)
            current_landmarks = pre_processed_landmark_list # Guardamos para usar si se pulsa tecla

            mp_drawing.draw_landmarks(
                debug_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # --- INTERFAZ DE USUARIO (UI) ---
    # Panel de información
    cv2.rectangle(debug_image, (0,0), (640, 40), (0,0,0), -1) # Barra negra superior

    # --- CAPTURA DE TECLAS ---
    key = cv2.waitKey(10)

    # 27 es ESC
    if key == 27:
        break

    # Detectar a-z (ASCII 97 a 122)
    if 97 <= key <= 122 and current_landmarks is not None:
        index = key - 97 # 'a' (97) se convierte en índice 0

        # Guardar
        logging_csv(index, current_landmarks)

        # Actualizar contador
        class_counts[index] += 1

        # Feedback visual
        char_saved = ALPHABET[index].upper()
        count_saved = class_counts[index]
        last_saved_msg = f"GUARDADO: '{char_saved}' | TOTAL: {count_saved}"
        last_saved_color = (0, 255, 0) # Verde

    # Mostrar mensaje en pantalla
    cv2.putText(debug_image, last_saved_msg, (10, 28),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, last_saved_color, 2, cv2.LINE_AA)

    # Mostrar instrucciones si no se ha guardado nada reciente
    if last_saved_msg == "Esperando...":
         cv2.putText(debug_image, "Pulsa a-z para capturar", (400, 28),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow('Sign-Lingo Collector (A-Z)', debug_image)

cap.release()
cv2.destroyAllWindows()
