import pandas as pd
import numpy as np

def generate_use_case():
    """
    Genera un DataFrame con valores atípicos para probar el escalado robusto.
    """
    np.random.seed(42)
    rows = 20
    df = pd.DataFrame({
        'feature_1': np.random.normal(100, 10, rows),
        'feature_2': np.random.uniform(0, 50, rows)
    })
    
    # Insertamos outliers agresivos:
    # Si usáramos Min-Max, estos valores comprimirían el resto de los datos a casi 0.
    df.iloc[0, 0] = 5000.0 
    df.iloc[1, 1] = -1000.0
    
    return df

def escalar_caracteristicas_robustas(df):
    """
    Transforma los datos para que tengan mediana 0 e IQR 1.
    Retorna un np.ndarray.
    """
    # 1. Calculamos la mediana de cada columna
    mediana = df.median()
    
    # 2. Calculamos el Rango Intercuartílico (IQR = Q3 - Q1)
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    
    # 3. Aplicamos la transformación robusta
    # La resta y división se alinean automáticamente por nombre de columna en Pandas
    df_scaled = (df - mediana) / iqr
    
    # 4. Convertimos el resultado a una matriz de NumPy
    return df_scaled.to_numpy()

# --- Bloque de ejecución para prueba local ---
if __name__ == "__main__":
    # Generamos los datos
    datos_originales = generate_use_case()
    
    # Aplicamos tu función
    resultado = escalar_caracteristicas_robustas(datos_originales)
    
    print("--- DataFrame Original (primeras 5 filas) ---")
    print(datos_originales.head())
    
    print("\n--- Matriz Escalada (primeras 5 filas) ---")
    print(resultado[:5])
    
    # Verificación rápida: La mediana de las columnas debería ser 0
    print("\nMedianas del resultado (deberían ser casi 0):")
    print(np.median(resultado, axis=0))
