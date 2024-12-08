import cv2
import mediapipe as mp
import os
import numpy as np
from tkinter import Tk, Label, Button, StringVar


# Configuración de MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

# Configuración del directorio de dataset
output_dir = "landmarks_dataset"
os.makedirs(output_dir, exist_ok=True)

# Añadimos la "Ñ" al alfabeto
letters = list("ABCDEFGHIJKLMNÑOPQRSTUVWXYZ")

# Variables globales
cap = None
current_letter_index = 0
capture_count = 200  # Número de capturas por letra


# Función para guardar landmarks
def guardar_landmarks(letter, landmarks):
    """
    Guarda los landmarks en un archivo .npy correspondiente a la letra actual.
    """
    output_path = os.path.join(output_dir, f"{letter}.npy")
    if os.path.exists(output_path):
        # Si ya existe el archivo, cargamos y añadimos los nuevos landmarks
        data = np.load(output_path)
        data = np.vstack([data, landmarks])
    else:
        # Si no existe, creamos un nuevo archivo
        data = np.array(landmarks)
    np.save(output_path, data)


def capturar_landmarks():
    """
    Captura landmarks para la letra seleccionada.
    """
    global cap, current_letter_index, capture_count
    if not cap:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error al abrir la cámara.")
            return

    count = 0
    letter = letters[current_letter_index]
    capture_count = 20  # Reiniciar contador para la letra actual

    while count < capture_count:
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar el frame.")
            break

        # Procesar la imagen con MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extraer coordenadas de landmarks
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                # Guardar los landmarks
                guardar_landmarks(letter, landmarks)
                count += 1
                print(f"Landmarks guardados para letra {letter}: {count}/{capture_count}.")

        # Mostrar letra y capturas restantes en el frame
        cv2.putText(frame, f"Letra actual: {letter}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Capturas restantes: {capture_count - count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Mostrar el frame
        cv2.imshow("Captura de Landmarks", frame)

        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"Captura completada para la letra {letter}.")
    cerrar_camara()


def cerrar_camara():
    """
    Libera la cámara y cierra cualquier ventana de OpenCV.
    """
    global cap
    if cap:
        cap.release()
        cv2.destroyAllWindows()
        cap = None


def avanzar_letra():
    """
    Cambia a la siguiente letra.
    """
    global current_letter_index, capture_count
    current_letter_index = (current_letter_index + 1) % len(letters)
    capture_count = 20  # Reiniciar contador
    actualizar_letra()


def retroceder_letra():
    """
    Cambia a la letra anterior.
    """
    global current_letter_index, capture_count
    current_letter_index = (current_letter_index - 1) % len(letters)
    capture_count = 20  # Reiniciar contador
    actualizar_letra()


def actualizar_letra():
    """
    Actualiza la etiqueta de la letra actual en la interfaz.
    """
    letra_var.set(f"Letra actual: {letters[current_letter_index]}")


def iniciar_interfaz():
    """
    Inicia la interfaz gráfica con Tkinter.
    """
    # Configuración de Tkinter
    root = Tk()
    root.title("Captura de Dataset - Landmarks")
    root.geometry("800x600")  # Ventana de tamaño grande

    global letra_var
    letra_var = StringVar()
    letra_var.set(f"Letra actual: {letters[current_letter_index]}")

    # Etiqueta para mostrar la letra actual
    label = Label(root, textvariable=letra_var, font=("Arial", 24))
    label.pack(pady=20)

    # Botón para capturar landmarks
    btn_capturar = Button(root, text="Capturar Landmarks", command=capturar_landmarks, font=("Arial", 16), width=20)
    btn_capturar.pack(pady=10)

    # Botón para avanzar a la siguiente letra
    btn_avanzar = Button(root, text="Siguiente Letra", command=avanzar_letra, font=("Arial", 16), width=20)
    btn_avanzar.pack(pady=10)

    # Botón para retroceder a la letra anterior
    btn_retroceder = Button(root, text="Letra Anterior", command=retroceder_letra, font=("Arial", 16), width=20)
    btn_retroceder.pack(pady=10)

    # Botón para salir y cerrar cámara
    btn_salir = Button(root, text="Salir", command=lambda: [cerrar_camara(), root.destroy()], font=("Arial", 16), width=20)
    btn_salir.pack(pady=20)

    root.mainloop()


if __name__ == "__main__":
    iniciar_interfaz()

