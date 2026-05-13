import cv2
import numpy as np
import os

# Configuración de rutas
INPUT_PATH = 'data/raw/'
OUTPUT_PATH = 'data/processed/'
IMG_SIZE = 128

def preparar_datos():
    lista_vectores = []
    
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    for filename in os.listdir(INPUT_PATH):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            # 1. Leer imagen
            img = cv2.imread(os.path.join(INPUT_PATH, filename))
            
            # 2. Convertir a escala de grises (Blanco y Negro)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 3. Redimensionar a 128x128
            resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
            
            # 4. Binarización (Umbralización)
            # El profesor pide 1 para blanco y 0 para objeto.
            # Usamos un umbral (threshold) para separar el fondo blanco del objeto.
            _, binarizada = cv2.threshold(resized, 200, 255, cv2.THRESH_BINARY_INV)
            
            # Convertir 255 a 1 y mantener 0 como 0
            matriz_final = (binarizada / 255).astype(int)
            
            # 5. Convertir a vector fila
            vector_fila = matriz_final.flatten()
            
            # 6. Etiquetado (Labeling)
            # Si el nombre del archivo contiene 'arroz', etiqueta 1, si no 0
            etiqueta = 1 if 'arroz' in filename.lower() else 0
            vector_con_etiqueta = np.append(vector_fila, etiqueta)
            
            lista_vectores.append(vector_con_etiqueta)
            
            # Guardar imagen procesada para revisión visual
            cv2.imwrite(os.path.join(OUTPUT_PATH, f"proc_{filename}"), binarizada)

    # 7. Crear matriz final del grupo (Dataset)
    dataset = np.array(lista_vectores)
    np.savetxt("dataset_daniel_valverde.csv", dataset, delimiter=",", fmt='%d')
    print(f"Procesamiento completado. Archivo 'dataset_daniel_valverde.csv' generado con {len(lista_vectores)} muestras.")

if __name__ == "__main__":
    preparar_datos()