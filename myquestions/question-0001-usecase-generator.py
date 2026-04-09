import pandas as pd
import numpy as np

def generar_caso_de_uso_escalar_caracteristicas_robustas():
    rows = np.random.randint(15, 30)
    df = pd.DataFrame({
        'f1': np.random.normal(100, 10, rows),
        'f2': np.random.uniform(0, 50, rows)
    })
    # Outliers aleatorios
    df.iloc[0, 0] = np.random.choice([5000.0, -5000.0])
    return df

def escalar_caracteristicas_robustas(df):
    mediana = df.median()
    iqr = df.quantile(0.75) - df.quantile(0.25)
    return ((df - mediana) / iqr).to_numpy()
    print(resultado[:5])
    
    # Verificación rápida: La mediana de las columnas debería ser 0
    print("\nMedianas del resultado (deberían ser casi 0):")
    print(np.median(resultado, axis=0))
