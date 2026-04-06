import numpy as np

def generate_usecase():
    """Genera vectores aleatorios de etiquetas reales y predicciones."""
    size = np.random.randint(50, 100)
    y_real = np.random.randint(0, 2, size)
    y_pred = np.random.randint(0, 2, size)
    return y_real, y_pred

# Ejemplo de uso:
# y_true, y_predicted = generate_usecase()
