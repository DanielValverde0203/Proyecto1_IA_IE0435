import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# 1. Cargar y dividir los datos
df = pd.read_csv("dataset_final_consolidado.csv", header=None)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Definir los algoritmos
modelos = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    "SVM_Lineal": SVC(kernel='linear'),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Regresion_Logistica": LogisticRegression(max_iter=1000),
    "Red_Neuronal": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
}

# 3. Entrenar y exportar cada modelo
print("Iniciando entrenamiento y exportación...\n")

for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    predicciones = modelo.predict(X_test)
    exactitud = accuracy_score(y_test, predicciones)
    print(f"Modelo {nombre} entrenado (Exactitud: {exactitud*100:.1f}%)")
    
    nombre_archivo = f"{nombre}.joblib"
    joblib.dump(modelo, nombre_archivo)
    print(f" -> Guardado como '{nombre_archivo}'\n")

print("¡Proceso completado! Los archivos .joblib están listos.")
