import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_shap_values(model, X_test):
    """
    Calcola i valori SHAP e restituisce un DataFrame con importanza media
    e range delle feature originali (non SHAP).
    
    Args:
        model: modello tree-based addestrato (es. XGBoost, LightGBM)
        X_test (pd.DataFrame): Dati di test

    Returns:
        pd.DataFrame: Importanza media SHAP e range reale delle feature
        shap_values: valori SHAP grezzi
        explainer: oggetto SHAP explainer
    """
    if not isinstance(X_test, pd.DataFrame):
        raise ValueError("X_test deve essere un DataFrame")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Supporta modelli multiclass e binari
    if isinstance(shap_values, list):  # Multiclass
        mean_abs = [np.mean(np.abs(sv), axis=0) for sv in shap_values]
        importance_df = pd.DataFrame(mean_abs).T
        importance_df.columns = [f'class_{i}' for i in range(len(shap_values))]
    else:  # Binario o regressione
        importance_df = pd.DataFrame({'importance': np.mean(np.abs(shap_values), axis=0)})

    importance_df['feature'] = X_test.columns
    importance_df['range'] = [
        f"{X_test[col].min():.2f} → {X_test[col].max():.2f}" for col in X_test.columns
    ]

    return importance_df.set_index('feature'), shap_values, explainer


def plot_interaction_summary(model, X_test, max_display=10):
    """
    Plotta il summary plot delle interazioni SHAP.

    Args:
        model: modello tree-based
        X_test (pd.DataFrame): Dati di test
        max_display (int): Numero massimo di feature da visualizzare
    """
    explainer = shap.TreeExplainer(model)
    interaction_values = explainer.shap_interaction_values(X_test)

    if isinstance(interaction_values, list):  # Multiclass
        for i, val in enumerate(interaction_values):
            print(f"\n🔍 Interazioni – Classe {i}")
            shap.summary_plot(val, X_test, plot_type="dot", max_display=max_display)
    else:
        shap.summary_plot(interaction_values, X_test, plot_type="dot", max_display=max_display)


def plot_shap_boxplot(shap_values, X_test, class_idx=0, top_n=5):
    """
    Plotta boxplot dei valori SHAP per le top feature più importanti.

    Args:
        shap_values: Output di SHAP (list o array)
        X_test (pd.DataFrame): Dati di test
        class_idx (int): Indice classe da analizzare (se multiclass)
        top_n (int): Numero di feature da visualizzare
    """
    if isinstance(shap_values, list):
        shap_array = shap_values[class_idx]
        title = f"SHAP Distribution – Classe {class_idx}"
    else:
        shap_array = shap_values
        title = "SHAP Distribution – Binario o Regressione"

    shap_array = np.array(shap_array)
    if shap_array.ndim > 2:
        shap_array = shap_array.reshape(shap_array.shape[0], -1)

    shap_df = pd.DataFrame(shap_array, columns=X_test.columns)
    top_features = shap_df.abs().mean().sort_values(ascending=False).head(top_n).index

    melted = shap_df[top_features].melt(var_name='Feature', value_name='SHAP Value')

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='SHAP Value', y='Feature', data=melted, palette='vlag')
    plt.axvline(0, color='black', linestyle='--', linewidth=1)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def generate_narrative_report(shap_df, class_name="la condizione predetta"):
    """
    Genera spiegazioni automatiche in linguaggio naturale per ciascuna feature.

    Args:
        shap_df (pd.DataFrame): Output da get_shap_values (index=feature, colonne: importance, range, ecc.)
        class_name (str): Etichetta della classe predetta (es. "caduta", "stadio avanzato")

    Returns:
        List[str]: Frasi interpretative per ciascuna feature
    """
    sentences = []
    for feature, row in shap_df.iterrows():
        imp = row.filter(like='importance').values[0]
        rng = row['range']

        if imp > 0.1:
            strength = "fortemente"
        elif imp > 0.05:
            strength = "moderatamente"
        else:
            strength = "lievemente"

        sentence = (
            f"📌 La variabile '{feature}' con range osservato {rng} "
            f"contribuisce {strength} alla previsione di {class_name} secondo il modello SHAP."
        )
        sentences.append(sentence)

    return sentences
