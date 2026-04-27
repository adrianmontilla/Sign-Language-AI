import cv2
import mediapipe as mp
import pickle
import numpy as np
import copy
import random
import time
import os
from collections import deque
from PIL import Image, ImageFont, ImageDraw

# Intentamos importar TensorFlow
try:
    import tensorflow as tf
except ImportError:
    print("⚠️ ADVERTENCIA: TensorFlow no está instalado.")
    tf = None

# ==========================================
# 1. CONFIGURACIÓN Y CARGA DE MODELOS
# ==========================================

# --- A. MODELO ESTÁTICO ---
try:
    with open('./RandomForestEstaticasPickle_model.pkl', 'rb') as f:
        model_static = pickle.load(f)
    print("✅ Modelo Estático cargado.")
except:
    print("⚠️ No se encuentra 'model.p'. (Solo funcionarán las letras dinámicas)")
    model_static = None

# --- B. MODELO DINÁMICO ---
LSTM_FILENAMES = ['./modelo_dinamico.h5', './mejor_modelo.keras', './model_lstm.keras']
model_dynamic = None

if tf:
    for name in LSTM_FILENAMES:
        if os.path.exists(name):
            try:
                model_dynamic = tf.keras.models.load_model(name)
                print(f"✅ Modelo Dinámico cargado: {name}")
                break
            except Exception as e:
                pass

# ==========================================
# 2. DEFINICIÓN DE CLASES Y REGLAS
# ==========================================

SEQUENCE_LENGTH = 30 
CONFIDENCE_THRESHOLD = 0.70
WIN_SCORE = 15  # <--- PUNTUACIÓN OBJETIVO PARA GANAR

STATIC_CLASSES_LIST = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
    'U', 'W', 'CH', 'Ñ' 
]
DYNAMIC_CLASSES_LIST = ['V', 'X', 'Y', 'Z', 'LL']

# --- FUSIÓN DE LISTAS ---
TARGET_LETTERS = []
if model_static:
    TARGET_LETTERS += STATIC_CLASSES_LIST
if model_dynamic:
    TARGET_LETTERS += DYNAMIC_CLASSES_LIST
TARGET_LETTERS = sorted(list(set(TARGET_LETTERS)))

if not TARGET_LETTERS:
    TARGET_LETTERS = ['A', 'B'] 

# Configuración Visual
TIME_BEFORE_HINT =8
HINT_DURATION = 20
DATA_FOLDER = './data'

# Colores (R, G, B) para Pillow
COLOR_BG_PANEL = (35, 35, 40)
COLOR_ACCENT = (255, 191, 0)      
COLOR_TEXT_MAIN = (245, 245, 245)
COLOR_SUCCESS = (0, 255, 127)     
COLOR_WARNING = (255, 215, 0)     
COLOR_ERROR = (255, 80, 80)       

# ==========================================
# 3. FUNCIONES DE UTILIDAD
# ==========================================

def put_text_pil(img, text, position, font_size, color, anchor="mm"):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except:
            font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color, anchor=anchor)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]
    for index, landmark_point in enumerate(temp_landmark_list):
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    flatten_list = []
    for temp_point in temp_landmark_list:
        flatten_list.append(temp_point[0])
        flatten_list.append(temp_point[1])
    max_value = max(list(map(abs, flatten_list)))
    def normalize_(n):
        return n / max_value if max_value != 0 else 0
    flatten_list = list(map(normalize_, flatten_list))
    return flatten_list

def smart_resize_fixed(image, width, height):
    h, w = image.shape[:2]
    scale = min(width/w, height/h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    y_off = (height - new_h) // 2
    x_off = (width - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    return canvas

def overlay_transparent_rect(img, x, y, w, h, color, alpha=0.5):
    sub_img = img[y:y+h, x:x+w]
    rect = np.full(sub_img.shape, color, dtype=np.uint8)
    res = cv2.addWeighted(sub_img, 1-alpha, rect, alpha, 0)
    img[y:y+h, x:x+w] = res

def get_confidence_color(score):
    if score >= CONFIDENCE_THRESHOLD:
        return COLOR_SUCCESS
    elif score >= 0.4:
        return COLOR_WARNING
    else:
        return COLOR_ERROR

# Cargar imágenes de ayuda
ref_images = {}
print("📂 Cargando imágenes...")
for letter in TARGET_LETTERS:
    filename = f"img_{letter}.png"
    path = os.path.join(DATA_FOLDER, filename)
    img = cv2.imread(path)
    if img is None:
        path = os.path.join(DATA_FOLDER, f"img_{letter}.jpg")
        img = cv2.imread(path)
    ref_images[letter] = img 

# ==========================================
# 4. INICIALIZACIÓN
# ==========================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

drawing_spec_points = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=3, circle_radius=3)
accent_bgr = (COLOR_ACCENT[2], COLOR_ACCENT[1], COLOR_ACCENT[0]) 
drawing_spec_lines = mp_drawing.DrawingSpec(color=accent_bgr, thickness=2, circle_radius=2)

sequence_history = deque(maxlen=SEQUENCE_LENGTH)
current_target = random.choice(TARGET_LETTERS)
score = 0
last_success_time = 0
wait_time = 2.0
success_mode = False
attempt_start_time = time.time()
hint_active = False
hint_start_time = 0
game_won = False # Nuevo estado: Victoria

print(f">>> JUEGO LISTO. Objetivo: {WIN_SCORE} Puntos.")

# ==========================================
# 5. BUCLE PRINCIPAL
# ==========================================
while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape
    
    # --- PROCESAMIENTO DE TECLAS ---
    key = cv2.waitKey(1) & 0xFF
    if key == 27: # ESC
        break
    
    # Solo procesamos teclas de juego si NO hemos ganado aún
    if not game_won:
        # Tecla S: SALTAR LETRA (SKIP)
        if key == ord('s') or key == ord('S'):
            print("⏩ Saltando letra...")
            current_target = random.choice(TARGET_LETTERS)
            attempt_start_time = time.time()
            hint_active = False
            sequence_history.clear()
            success_mode = False
    
    # Tecla R: REINICIAR (Funciona siempre, incluso en victoria)
    if key == ord('r') or key == ord('R'):
        print("🔄 Reiniciando juego...")
        score = 0
        game_won = False
        current_target = random.choice(TARGET_LETTERS)
        attempt_start_time = time.time()
        hint_active = False
        sequence_history.clear()
        success_mode = False

    # ----------------------------------------------
    # LÓGICA DE JUEGO
    # ----------------------------------------------
    if not game_won:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        prediction_char = "?"
        confidence_score = 0.0
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, drawing_spec_points, drawing_spec_lines)

                # Extracción de datos
                landmark_list = []
                for landmark in hand_landmarks.landmark:
                    landmark_list.append([landmark.x, landmark.y])
                processed = pre_process_landmark(landmark_list)
                sequence_history.append(processed)
                
                # Inferencia
                current_pred_char = "?"
                current_conf = 0.0

                if current_target in DYNAMIC_CLASSES_LIST:
                    if model_dynamic is not None and len(sequence_history) == SEQUENCE_LENGTH:
                        input_data = np.array([sequence_history])
                        try:
                            pred = model_dynamic.predict(input_data, verbose=0)[0]
                            idx = np.argmax(pred)
                            current_pred_char = DYNAMIC_CLASSES_LIST[idx]
                            current_conf = pred[idx]
                        except: pass
                else:
                    if model_static:
                        try:
                            pred = model_static.predict_proba([processed])[0]
                            idx = np.argmax(pred)
                            if idx < len(STATIC_CLASSES_LIST):
                                current_pred_char = STATIC_CLASSES_LIST[idx]
                                current_conf = pred[idx]
                        except: pass
                
                if current_conf > 0.2:
                    prediction_char = current_pred_char
                    confidence_score = current_conf
                else:
                    prediction_char = "..."

                # Comprobar acierto
                if not success_mode and prediction_char == current_target and confidence_score >= CONFIDENCE_THRESHOLD:
                    success_mode = True
                    last_success_time = time.time()
                    score += 1
                    
                    # CHECK VICTORIA
                    if score >= WIN_SCORE:
                        game_won = True
                        last_success_time = time.time() # Para que se vea el ultimo frame un momento
                    
                    hint_active = False

        else:
            if len(sequence_history) > 0: sequence_history.clear()

        # Ayuda y Turnos
        if not success_mode:
            elapsed = time.time() - attempt_start_time
            if elapsed > TIME_BEFORE_HINT and not hint_active:
                hint_active = True
                hint_start_time = time.time()
            if hint_active and (time.time() - hint_start_time) > HINT_DURATION:
                hint_active = False
                attempt_start_time = time.time()

        if success_mode and not game_won and (time.time() - last_success_time > wait_time):
            success_mode = False
            current_target = random.choice(TARGET_LETTERS)
            attempt_start_time = time.time()
            hint_active = False
            sequence_history.clear()

    # ==========================================
    # DIBUJADO DE INTERFAZ
    # ==========================================
    
    # Si ganamos, mostramos pantalla de victoria
    if game_won:
        # Filtro oscuro sobre todo
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (W, H), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Textos de Victoria
        cx, cy = W // 2, H // 2
        frame = put_text_pil(frame, "¡FELICIDADES!", (cx, cy - 80), 80, COLOR_SUCCESS)
        frame = put_text_pil(frame, f"Completaste los {WIN_SCORE} puntos", (cx, cy + 20), 40, (255,255,255))
        frame = put_text_pil(frame, "[R] Reiniciar   [ESC] Salir", (cx, cy + 100), 25, (180,180,180))

    else:
        # INTERFAZ NORMAL DE JUEGO
        panel_w = int(W * 0.30)
        panel_x = W - panel_w
        
        bg_panel_bgr = (COLOR_BG_PANEL[2], COLOR_BG_PANEL[1], COLOR_BG_PANEL[0])
        overlay_transparent_rect(frame, panel_x, 0, panel_w, H, bg_panel_bgr, alpha=0.85)
        cv2.line(frame, (panel_x, 0), (panel_x, H), accent_bgr, 2)

        # Header y Objetivo
        cy_header = int(H * 0.15)
        frame = put_text_pil(frame, "REALIZA:", (panel_x + 20, int(H * 0.08)), 25, (180,180,180), anchor="ls")
        
        color_target = COLOR_SUCCESS if success_mode else COLOR_ACCENT
        if success_mode and int(time.time()*10) % 2 == 0: color_target = (200, 255, 200)
        target_size = 120 if W > 1000 else 80
        frame = put_text_pil(frame, current_target, (panel_x + panel_w//2, cy_header + 40), target_size, color_target)

        # Feedback
        cy_feedback = int(H * 0.45)
        frame = put_text_pil(frame, "DETECTADO:", (panel_x + 20, cy_feedback), 22, (180,180,180), anchor="ls")
        pred_color = get_confidence_color(confidence_score)
        pred_text = prediction_char if prediction_char != "?" else "..."
        frame = put_text_pil(frame, pred_text, (panel_x + panel_w//2, cy_feedback + 60), 60, pred_color)
        
        # Barra
        bar_w = int(panel_w * 0.8)
        bar_h = 15
        bar_x = panel_x + (panel_w - bar_w) // 2
        bar_y = cy_feedback + 100
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60,60,60), -1)
        fill_w = int(bar_w * confidence_score)
        pred_color_bgr = (pred_color[2], pred_color[1], pred_color[0])
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), pred_color_bgr, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200,200,200), 1)
        frame = put_text_pil(frame, f"{int(confidence_score*100)}%", (bar_x + bar_w + 35, bar_y + 7), 18, (200,200,200))

        # Puntos y Controles
        frame = put_text_pil(frame, f"META: {score}/{WIN_SCORE}", (panel_x + 20, H - 90), 30, COLOR_WARNING, anchor="ls")
        
        # Instrucciones de teclas (NUEVO)
        controls_text = "[S] Saltar letra   [ESC] Salir"
        frame = put_text_pil(frame, controls_text, (panel_x + 20, H - 50), 16, (150,150,150), anchor="ls")

        # Overlay Acierto
        if success_mode:
            cam_center_x = (W - panel_w) // 2
            cam_center_y = H // 2
            frame = put_text_pil(frame, "¡MUY BIEN!", (cam_center_x + 3, cam_center_y + 3), 60, (0,0,0))
            frame = put_text_pil(frame, "¡MUY BIEN!", (cam_center_x, cam_center_y), 60, COLOR_SUCCESS)

        # Ayuda
        if hint_active and not success_mode:
            help_img = ref_images.get(current_target)
            if help_img is not None:
                hint_w = int((W - panel_w) * 0.3)
                hint_h = int(hint_w)
                hx = 20
                hy = H - hint_h - 20
                overlay_transparent_rect(frame, hx-10, hy-30, hint_w+20, hint_h+40, (0,0,0), 0.7)
                frame = put_text_pil(frame, "AYUDA", (hx + 100, hy - 7), 20, COLOR_ACCENT, anchor="ls")
                resized_help = smart_resize_fixed(help_img, hint_w, hint_h)
                frame[hy:hy+hint_h, hx:hx+hint_w] = resized_help
                cv2.rectangle(frame, (hx, hy), (hx+hint_w, hy+hint_h), accent_bgr, 2)

    cv2.imshow('SingLingo - Entrenamiento', frame)

cap.release()
cv2.destroyAllWindows()
