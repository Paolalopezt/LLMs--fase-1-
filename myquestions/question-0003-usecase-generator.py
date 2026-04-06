import pandas as pd
import numpy as np

def generate_usecase():
    rows = 50
    target = np.random.rand(rows)
    df = pd.DataFrame({
        'feat1': target + np.random.normal(0, 0.1, rows),
        'feat2': np.random.rand(rows),
        'target': target
    })
    return df, 'target', 1
