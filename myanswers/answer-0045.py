import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

def seleccion_y_validacion(df, target_col, k_features):
    X = df.drop(columns=[target_col]).to_numpy()
    y = df[target_col].to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    selector = SelectKBest(score_func=mutual_info_classif, k=k_features)
    selector.fit(X_scaled, y)
    selected_indices = np.where(selector.get_support())[0].tolist()
    X_selected = selector.transform(X_scaled)
    model = GradientBoostingClassifier(random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_selected, y, cv=cv, scoring="accuracy")
    return (selected_indices, float(np.mean(scores)), (int(X.shape[1]), int(k_features)))
