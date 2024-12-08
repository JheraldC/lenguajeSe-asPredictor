import os
import numpy as np

# Ruta donde se guardan los archivos .npy
dataset_dir = "landmarks_dataset"

# Añadimos la "Ñ" al alfabeto
letters = list("ABCDEFGHIJKLMNÑOPQRSTUVWXYZ")

X = []  # Landmarks
y = []  # Etiquetas

# Cargar landmarks de cada archivo y asignar etiquetas
for letter in letters:
    file_path = os.path.join(dataset_dir, f"{letter}.npy")
    if os.path.exists(file_path):
        data = np.load(file_path)  # Cargar landmarks de la letra
        if data.ndim == 2 and data.shape[1] == 63:  # Validar formato esperado
            X.append(data)  # Agregar los landmarks al conjunto
            y.extend([letter] * len(data))  # Agregar etiquetas correspondientes
        else:
            print(f"Error: El archivo '{file_path}' tiene un formato inesperado (esperado: 2D array con 63 columnas).")
    else:
        print(f"Advertencia: No se encontró un archivo para la letra '{letter}'.")

# Verificar que se encontraron datos
if not X:
    print("Error: No se encontraron datos válidos en el directorio.")
    exit()

# Convertir a arrays de NumPy
X = np.vstack(X)  # Combina todas las muestras en una sola matriz
y = np.array(y)   # Convertir etiquetas a array

# Guardar el conjunto de datos combinado
np.save("landmarks_combined.npy", X)
np.save("labels_combined.npy", y)

print("Datos combinados guardados en landmarks_combined.npy y labels_combined.npy.")
print(f"Total de muestras: {len(X)}")
print(f"Letras incluidas: {np.unique(y)}")

