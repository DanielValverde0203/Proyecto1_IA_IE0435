import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# 1. Cargar el dataset consolidado
df = pd.read_csv("dataset_final_consolidado.csv", header=None)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# 2. Dividir en Entrenamiento (80%) y Prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Definir el catálogo de algoritmos a probar
modelos = {
    "SVM Lineal": SVC(kernel='linear'),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
    "Regresión Logística": LogisticRegression(max_iter=1000),
    "Red Neuronal (Perceptrón Multicapa)": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
}

# 4. Iterar, entrenar y evaluar cada algoritmo
print("Iniciando evaluación comparativa de algoritmos...\n")

for nombre, modelo in modelos.items():
    print(f"================ {nombre} ================")
    
    # Entrenar el modelo
    modelo.fit(X_train, y_train)
    
    # Realizar predicciones con los datos de prueba
    predicciones = modelo.predict(X_test)
    
    # Calcular y mostrar métricas
    exactitud = accuracy_score(y_test, predicciones)
    print(f"Exactitud (Accuracy): {exactitud:.4f}")
    print("Matriz de Confusión:")
    print(confusion_matrix(y_test, predicciones))
    print("\n")
