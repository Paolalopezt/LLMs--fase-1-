import pandas as pd
import numpy as np

def generar_caso_de_uso_seleccionar_top_correlacion():
    """
    Genera un DataFrame con variables de distinta correlación respecto al target.
    """
    rows = np.random.randint(40, 60)
    target = np.random.rand(rows)
    
    # feat1 tendrá correlación muy alta (target + ruido mínimo)
    # feat2 será puro ruido (correlación baja)
    # feat3 tendrá correlación negativa fuerte
    df = pd.DataFrame({
        'feat1': target + np.random.normal(0, 0.01, rows),
        'feat2': np.random.rand(rows),
        'feat3': -target + np.random.normal(0, 0.01, rows),
        'target': target
    })
    
    target_col = 'target'
    k = 2
    return df, target_col, k

def seleccionar_top_correlacion(df, target_col, k):
    """
    Identifica las k columnas con mayor correlación absoluta con la columna objetivo.
    """
    # 1. Calculamos la matriz de correlación de Pearson
    correlaciones = df.corr()
    
    # 2. Extraemos la correlación respecto al target (excluyendo el target mismo)
    target_corr = correlaciones[target_col].drop(labels=[target_col])
    
    # 3. Tomamos el valor absoluto (para incluir correlaciones negativas fuertes)
    abs_corr = target_corr.abs()
    
    # 4. Ordenamos de mayor a menor y tomamos las primeras k
    top_k = abs_corr.sort_values(ascending=False).head(k)
    
    # 5. Retornamos los nombres de las columnas como np.ndarray
    return top_k.index.to_numpy()
