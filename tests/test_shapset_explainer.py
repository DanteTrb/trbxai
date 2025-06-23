from trbxai.explain.shapset_explainer import get_shap_values
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def test_get_shap_values():
    X = pd.DataFrame({
        "f1": [1, 2, 3],
        "f2": [4, 5, 6]
    })
    y = [0, 1, 0]
    model = RandomForestClassifier().fit(X, y)

    shap_vals = get_shap_values(model, X)

    assert shap_vals is not None