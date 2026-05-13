import pandas as pd

# 1. Arreglar archivo dataset_caleb
print("Procesando archivo dataset caleb...")
df_caleb = pd.read_csv('dataset_caleb.csv', header=None)
df_caleb[16384] = [1] * 15 + [0] * 15 

# 2. Arreglar dataset_felipe
print("Procesando archivo dataset felipe...")
df_felipe = pd.read_csv('dataset_felipe.csv', skiprows=1, header=None)

# 3. Arreglar archivo dataset_cristopher
print("Procesando archivo dataset cristopher...")
df_cris = pd.read_csv('dataset_cristopher.csv', skiprows=1, header=None)
pixeles_invertidos = 1 - df_cris.iloc[:, :-1]
df_cris_corregido = pd.concat([pixeles_invertidos, df_cris.iloc[:, -1]], axis=1)
df_cris_corregido.columns = range(16385)

# 4. Leer dataset de daniel
print("Procesando archivo de Daniel...")
df_daniel = pd.read_csv('dataset_daniel_valverde.csv', header=None)

# 5. Unir todo el dataset
print("Uniendo datasets...")
dataset_completo = pd.concat([df_caleb, df_felipe, df_cris_corregido, df_daniel], ignore_index=True)

# Mezclamos las 120 filas resultantes de forma aleatoria para el modelo
dataset_completo = dataset_completo.sample(frac=1, random_state=42).reset_index(drop=True)

# Guardar
dataset_completo.to_csv("dataset_final_consolidado.csv", index=False, header=False)

print("\n¡Dataset conformado con éxito!")
print(f"Total de muestras (filas): {dataset_completo.shape[0]}")
print(f"Total de características + etiqueta: {dataset_completo.shape[1]}")
