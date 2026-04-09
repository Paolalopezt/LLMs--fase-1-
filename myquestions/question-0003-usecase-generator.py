import pandas as pd
import numpy as np

def generar_caso_de_uso_seleccionar_top_correlacion():
    rows = np.random.randint(40, 60)
    target = np.random.rand(rows)
    df = pd.DataFrame({
        'feat1': target + np.random.normal(0, 0.01, rows),
        'feat2': np.random.rand(rows),
        'target': target
    })
    # Empaquetamos los 3 argumentos en el diccionario
    args = {'df': df, 'target_col': 'target', 'k': 1}
    return args, None

def seleccionar_top_correlacion(df, target_col, k):
    correlaciones = df.corr()
    target_corr = correlaciones[target_col].drop(labels=[target_col])
    abs_corr = target_corr.abs()
    top_k = abs_corr.sort_values(ascending=False).head(k)
    return top_k.index.to_numpy()
