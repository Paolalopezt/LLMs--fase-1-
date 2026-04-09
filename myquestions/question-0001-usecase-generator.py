import pandas as pd
import numpy as np

def generar_caso_de_uso_escalar_caracteristicas_robustas():
    rows = np.random.randint(15, 30)
    df = pd.DataFrame({
        'f1': np.random.normal(100, 10, rows),
        'f2': np.random.uniform(0, 50, rows)
    })
    df.iloc[0, 0] = 5000.0 
    # Retorna: (diccionario de argumentos, None como placeholder de respuesta)
    return {'df': df}, None

def escalar_caracteristicas_robustas(df):
    mediana = df.median()
    iqr = df.quantile(0.75) - df.quantile(0.25)
    return ((df - mediana) / iqr).to_numpy()
