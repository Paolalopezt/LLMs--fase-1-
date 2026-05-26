import numpy as np
from sklearn.cluster import KMeans

def generar_caso_de_uso_calcular_inercia_clusters():
    rows = np.random.randint(50, 100)
    X = np.random.rand(rows, 2)
    n_clusters = np.random.randint(2, 5)
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
    kmeans.fit(X)
    expected = float(kmeans.inertia_)
    return {'X': X, 'n_clusters': n_clusters}, expected

def calcular_inercia_clusters(X, n_clusters):
    pass
