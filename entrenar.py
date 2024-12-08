import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# Cargar los datos combinados
X = np.load("landmarks_combined.npy")
y = np.load("labels_combined.npy")

# Normalizar los landmarks (Min-Max scaling)
global_min = np.min(X, axis=0)
global_max = np.max(X, axis=0)
X = (X - global_min) / (global_max - global_min)

# Guardar los valores globales para normalización
np.save("global_min.npy", global_min)
np.save("global_max.npy", global_max)

# Codificar las etiquetas en formato one-hot
label_binarizer = LabelBinarizer()
y_encoded = label_binarizer.fit_transform(y)

# Dividir los datos en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Construir el modelo
model = models.Sequential([
    layers.Dense(256, activation='relu', input_shape=(63,)),  # Entrada con 63 coordenadas
    layers.Dropout(0.4),  # Regularización para evitar overfitting
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(len(y_encoded[0]), activation='softmax')  # Salida con tantas clases como etiquetas
])

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32)

# Guardar el modelo y las etiquetas
model.save("hand_sign_model_landmarks.h5")
np.save("class_labels.npy", label_binarizer.classes_)

print("Modelo entrenado y guardado correctamente.")

