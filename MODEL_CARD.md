# Model Card: Clasificador de Contaminantes

* **Model name + version:** Árbol de Decisión (Decision Tree) v1.0
* **Intended use:** Detección de anomalías en granos de arroz para control de calidad.
* **Data summary:** 120 imágenes binarizadas de 128x128 píxeles. Datos provenientes de 4 fuentes distintas con variaciones de luz y cámara.
* **Labeling process:** Etiquetado manual binario realizado por el equipo de trabajo.
* **Metrics:** Exactitud (Accuracy) del 79.2%, medida con un split de prueba del 20% estratificado.
* **Ethical/safety notes:** El modelo puede fallar ante cambios drásticos de fondo o iluminación extrema. No se recomienda para seguridad crítica sin supervisión humana.
* **Limitations:** Sensible a objetos muy pequeños y al desenfoque de la cámara.
* **Reproducibility:** Entrenado con la semilla `random_state=42` y `class_weight='balanced'`. Ejecutable mediante el script `scripts/exportar_modelos.py`.
