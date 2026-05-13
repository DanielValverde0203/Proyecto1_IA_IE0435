# Proyecto 1: Clasificación de Contaminantes en Líneas de Producción mediante IA

## Descripción General
Este repositorio documenta el diseño, desarrollo e implementación de un sistema de visión artificial y aprendizaje automático (*Machine Learning*) orientado al control de calidad en líneas de producción. El objetivo central del sistema es distinguir con alta precisión entre granos de arroz válidos (clase `1`) y elementos contaminantes como clips o aros metálicos (clase `0`), previniendo que objetos peligrosos lleguen al consumidor final.

El proyecto abarca desde la captura y preprocesamiento digital de las imágenes hasta la consolidación de un dataset colaborativo, el entrenamiento de múltiples algoritmos de clasificación y la exportación del modelo definitivo para su uso en producción.

---

## Fases del Proyecto

### 1. Preprocesamiento de Imágenes y Vectorización
Para estandarizar las fotografías tomadas en distintos entornos, se diseñó un flujo de trabajo utilizando OpenCV:
* **Redimensionamiento:** Escalado de todas las imágenes a una resolución estandarizada de $128 \times 128$ píxeles mediante interpolación digital.
* **Normalización y Binarización:** Conversión de las imágenes a escala de grises y posterior aplicación de un umbral para separar el objeto del fondo (Fondo = `1`, Objeto = `0`).
* **Vectorización (Flattening):** Aplanamiento de la matriz bidimensional de cada imagen en un vector unidimensional de $16,384$ características (píxeles), optimizado para la ingesta en algoritmos de clasificación clásica.

### 2. Consolidación y Limpieza del Dataset Grupal
Se recolectaron cuatro datasets aportados por los integrantes del grupo para generar un conjunto de datos robusto de 120 muestras. Durante la integración (`unir_datasets.py`), se aplicaron técnicas de ingeniería de datos para corregir inconsistencias:
* Inversión de matrices binarias ($1 - \text{píxel}$) en lotes que presentaban colores invertidos.
* Imputación de etiquetas faltantes.
* Eliminación de encabezados de texto y aleatorización (*shuffle*) de las filas para evitar sesgos durante el entrenamiento.
* **Resultado:** Un archivo maestro (`dataset_final_consolidado.csv`) con 120 filas y 16,385 columnas (16,384 características + 1 etiqueta).

### 3. Entrenamiento y Evaluación de Modelos
Se dividió el dataset consolidado utilizando un muestreo estratificado: 80% para entrenamiento (96 imágenes) y 20% para pruebas (24 imágenes). Se evaluaron cinco algoritmos clásicos utilizando `scikit-learn` para determinar la mejor aproximación matemática frente a la varianza de escala presente en las fotografías.

**Tabla Comparativa de Rendimiento:**

| Algoritmo | Exactitud (Accuracy) |
| :--- | :---: |
| **Random Forest** (Ganador) | 70.8% |
| **Regresión Logística** (Empate) | 70.8% |
| **SVM Lineal** | 62.5% |
| **K-Nearest Neighbors (KNN)** | 62.5% |
| **Red Neuronal (MLP)** | 54.2% |

**Análisis de Selección:** Tras evaluar los modelos, **Random Forest** y la **Regresión Logística** obtuvieron el mejor rendimiento general con una exactitud del 70.8%. Se seleccionó Random Forest como el algoritmo definitivo para exportar, ya que los modelos basados en árboles (ensambles) suelen manejar mejor la alta varianza y el ruido introducido por las severas diferencias de escala y encuadre en las imágenes aportadas por los distintos integrantes. El rendimiento moderado general demuestra que, para trabajos futuros, se debe establecer un protocolo físico de captura fotográfica más estricto.

---

## Estructura del Repositorio

* `dataset_final_consolidado.csv`: Dataset íntegro utilizado para el entrenamiento (120 muestras).
* `unir_datasets.py`: Script de estandarización, limpieza y concatenación de los archivos CSV originales.
* `evaluar_modelos.py`: Script para la evaluación comparativa de los 5 algoritmos propuestos.
* `exportar_modelos.py`: Script que entrena los algoritmos y exporta los objetos serializados.
* `RandomForest.joblib`: Modelo ganador pre-entrenado y listo para ser desplegado en producción.
* `resultados_modelos.md`: Reporte detallado de métricas y exactitud del proceso de entrenamiento.

---

## Uso del Modelo en Producción
Para implementar el clasificador en un entorno de pruebas con imágenes nuevas vectorizadas, se puede cargar el modelo serializado mediante la librería `joblib`.

---

## Autor
**Daniel Valverde Ortiz - C28041** *Escuela de Ingeniería Eléctrica* *Universidad de Costa Rica (UCR)*
