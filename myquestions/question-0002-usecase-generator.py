import numpy as np
from sklearn.metrics import precision_score

def generar_caso_de_uso_evaluar_precision_positiva():
    size = np.random.randint(50, 100)
    y_real = np.random.randint(0, 2, size)
    y_pred = np.random.randint(0, 2, size)
    # Retornamos el diccionario de argumentos y un None
    return {'y_real': y_real, 'y_pred': y_pred}, None

def evaluar_precision_positiva(y_real, y_pred):
    # Calculamos la precisión para la clase 1
    precision = precision_score(y_real, y_pred, pos_label=1, zero_division=0)
    return float(precision)
