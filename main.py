import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time

# Cargar el modelo y las clases
model = tf.keras.models.load_model("hand_sign_model_landmarks.h5")
class_labels = np.load("class_labels.npy")
global_min = np.load("global_min.npy")  # Valores globales para normalización
global_max = np.load("global_max.npy")

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Configuración de una única ventana de OpenCV
window_name = "Predicción de Lenguaje de Señas"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 800, 600)

# Configuración de la cámara
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se pudo acceder a la cámara.")
    exit()

print("Presiona 'q' para salir.")

prev_time = 0  # Para calcular el FPS
threshold = 0.7  # Umbral de confianza

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo capturar el frame.")
            break

        # Convertir frame a RGB para MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Si se detectan landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Dibujar landmarks en el frame
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extraer landmarks y normalizarlos
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])

                landmarks_array = np.array(landmarks).reshape(1, -1)
                landmarks_array = (landmarks_array - global_min) / (global_max - global_min)

                # Realizar la predicción
                prediction = model.predict(landmarks_array)
                predicted_label = class_labels[np.argmax(prediction)]
                probability = np.max(prediction)  # Probabilidad de la predicción

                # Mostrar la predicción o mensaje de "no reconocida"
                if probability > threshold:
                    cv2.putText(frame, f"Letra: {predicted_label} ({probability:.2f})", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Letra no reconocida", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Calcular y mostrar FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Mostrar el video en una única ventana
        cv2.imshow(window_name, frame)

        # Salir del bucle con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Liberar recursos de la cámara y cerrar ventanas
    cap.release()
    cv2.destroyAllWindows()

