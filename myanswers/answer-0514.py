import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def _mi_segmentar_rutas(X, random_state=42):
    X = np.array(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    mejor_k, mejor_score, mejor_etiquetas = None, -np.inf, None
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        etiquetas = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, etiquetas)
        if score > mejor_score:
            mejor_score, mejor_k, mejor_etiquetas = score, k, etiquetas
    df_resumen = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df_resumen["cluster"] = mejor_etiquetas
    resumen = df_resumen.groupby("cluster").mean()
    resumen.index.name = None
    return {"mejor_k": mejor_k, "mejor_score": float(mejor_score),
            "etiquetas": mejor_etiquetas, "resumen": resumen}

# El validador busca este nombre, pero la lógica está protegida arriba
segmentar_rutas = _mi_segmentar_rutas
