# Proyecto 1: Clasificación de Contaminaciones (IA) 

## Descripción
Este repositorio contiene el dataset y el código necesario para el procesamiento de imágenes orientado a la clasificación de objetos en líneas de producción. El sistema utiliza técnicas de visión artificial para identificar granos de arroz vs. elementos no deseados.

## Estructura del Proyecto
- `data/processed/`: Imágenes procesadas (binarizadas y normalizadas a 128x128).
- `dataset_daniel_valverde.csv`: Dataset final (vector de 16,384 píxeles por muestra).
- `procesar_imagenes.py`: Script en Python (OpenCV) para la automatización del preprocesamiento.

## Procedimiento Técnico
1. **Redimensionamiento:** Escalado de imágenes a 128x128 píxeles mediante interpolación digital.
2. **Normalización:** Conversión a escala de grises y binarización (fondo=1, objeto=0).
3. **Vectorización:** Aplanamiento de la matriz de píxeles para generar un CSV estandarizado.

## Autor
**Daniel Valverde Ortiz - C28041**
*Estudiante de Ingeniería Eléctrica - Universidad de Costa Rica*
