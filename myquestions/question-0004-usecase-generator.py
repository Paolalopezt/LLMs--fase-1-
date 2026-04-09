import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def generar_caso_de_uso_calcular_inercia_clusters():
    """
    Genera una matriz de datos aleatorios y un número de clusters.
    Incluye un componente aleatorio en dimensiones y número de centros.
    """
    # Tamaño aleatorio entre 50 y 150 filas, y 2 o 3 columnas
    rows = np.random.randint(50, 150)
    cols = np.random.randint(2, 4)
    X = np.random.rand(rows, cols)
    
    # Número de clusters aleatorio entre 2 y 5
    n_clusters = np.random.randint(2, 6)
    
    return X, n_clusters

def calcular_inercia_clusters(X, n_clusters):
    """
    Ejecuta K-Means y devuelve la inercia (WCSS).
    Utiliza sklearn.cluster.KMeans.
    """
    # Inicializamos el modelo con el número de clusters indicado
    # n_init='auto' es la recomendación en versiones recientes de sklearn
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
    
    # Entrenamos el modelo con los datos
    kmeans.fit(X)
    
    # Retornamos el atributo inertia_ que representa la suma de cuadrados internos
    return float(kmeans.inertia_)
