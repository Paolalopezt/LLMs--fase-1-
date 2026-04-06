import pandas as pd
import numpy as np

def generate_usecase():
    rows = 20
    df = pd.DataFrame({
        'feature_1': np.random.normal(100, 10, rows),
        'feature_2': np.random.uniform(0, 50, rows)
    })
    return df
