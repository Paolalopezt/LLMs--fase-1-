import pandas as pd

def resumen_estudiantes(df):
    return df.groupby('grupo')['nota'].mean().reset_index(name='promedio')
