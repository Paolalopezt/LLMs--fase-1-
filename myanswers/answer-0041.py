import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer

def preparar_datos(df, target_col):
    df_clean = df.interpolate().bfill()
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col].to_numpy()
    pt = PowerTransformer(method='yeo-johnson')
    X_trans = pt.fit_transform(X)
    return (X_trans, y)
