import numpy as np
from sklearn.metrics import precision_score

def generar_caso_de_uso_evaluar_precision_positiva():
    """
    Genera vectores aleatorios de etiquetas reales y predicciones.
    Cumple con el requisito de componente aleatorio en tamaño y valores.
    """
    size = np.random.randint(50, 150)
    # Generamos datos desbalanceados para simular el contexto de fraude
    y_real = np.random.choice([0, 1], size=size, p=[0.9, 0.1])
    y_pred = np.random.choice([0, 1], size=size, p=[0.8, 0.2])
    
    return y_real, y_pred

def evaluar_precision_positiva(y_real, y_pred):
    """
    Calcula la precisión (Precision) para la clase positiva (1).
    Utiliza sklearn.metrics.
    """
    # Calculamos la precisión específicamente para la clase 1
    # zero_division=0 evita errores si el modelo no predice ninguna clase positiva
    precision = precision_score(y_real, y_pred, pos_label=1, zero_division=0)
    
    return float(precision)
