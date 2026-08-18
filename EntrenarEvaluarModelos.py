import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from xgboost import XGBClassifier

INPUT_FILE = "dataset_features_ready.csv"
MODEL_RF_FILE = "modelo_random_forest.joblib"
MODEL_XGB_FILE = "modelo_xgboost.joblib"

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] No se encuentra el archivo '{INPUT_FILE}'.")
        return

    print(f"[*] Cargando dataset numérico desde '{INPUT_FILE}'...")
    df = pd.read_csv(INPUT_FILE)
    print(f"[+] Total de registros cargados: {len(df):,}")

    # 1. SEPARAR CARACTERÍSTICAS (X) Y ETIQUETAS (y)
    # Eliminamos 'sld' (texto) y 'label' para quedarnos solo con columnas numéricas
    features = [c for c in df.columns if c not in ['sld', 'label']]
    X = df[features]
    y = df['label']

    print(f"[*] Características utilizadas ({len(features)}): {features}")

    # 2. DIVISIÓN TRAIN / TEST (80% / 20%) ESTRATIFICADO
    print("[*] Dividiendo dataset en 80% Entrenamiento y 20% Pruebas...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"    [+] Set de Entrenamiento: {len(X_train):,} muestras")
    print(f"    [+] Set de Pruebas:       {len(X_test):,} muestras\n")

    # --------------------------------------------------------------------------
    # 3. ENTRENAMIENTO DE RANDOM FOREST
    # --------------------------------------------------------------------------
    print("="*65)
    print(" [1/2] ENTRENANDO RANDOM FOREST CLASSIFIER")
    print("="*65)
    t0 = time.time()
    
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        n_jobs=-1,  # Utiliza todos los núcleos de la CPU
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    t_rf = time.time() - t0
    print(f"[+] Entrenado en {t_rf:.2f} segundos.")

    # Evaluación RF
    y_pred_rf = rf_model.predict(X_test)
    y_proba_rf = rf_model.predict_proba(X_test)[:, 1]

    acc_rf = accuracy_score(y_test, y_pred_rf)
    auc_rf = roc_auc_score(y_test, y_proba_rf)
    print(f"[OK] Accuracy RF: {acc_rf*100:.2f}% | ROC-AUC: {auc_rf:.4f}\n")

    # --------------------------------------------------------------------------
    # 4. ENTRENAMIENTO DE XGBOOST
    # --------------------------------------------------------------------------
    print("="*65)
    print(" [2/2] ENTRENANDO XGBOOST CLASSIFIER")
    print("="*65)
    t0 = time.time()
    
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=10,
        learning_rate=0.1,
        n_jobs=-1,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    t_xgb = time.time() - t0
    print(f"[+] Entrenado en {t_xgb:.2f} segundos.")

    # Evaluación XGBoost
    y_pred_xgb = xgb_model.predict(X_test)
    y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    auc_xgb = roc_auc_score(y_test, y_proba_xgb)
    print(f"[OK] Accuracy XGBoost: {acc_xgb*100:.2f}% | ROC-AUC: {auc_xgb:.4f}\n")

    # --------------------------------------------------------------------------
    # 5. MATRICES DE CONFUSIÓN Y REPORTES DETALLADOS
    # --------------------------------------------------------------------------
    print("="*65)
    print(" REPORTES DE CLASIFICACIÓN Y MATRICES DE CONFUSIÓN")
    print("="*65)

    print("\n--- REPORT RANDOM FOREST ---")
    print(classification_report(y_test, y_pred_rf, target_names=['Benigno (0)', 'DGA (1)'], digits=4))

    print("\n--- REPORT XGBOOST ---")
    print(classification_report(y_test, y_pred_xgb, target_names=['Benigno (0)', 'DGA (1)'], digits=4))

    # Guardar Matrices de Confusión como Imagen
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cm_rf = confusion_matrix(y_test, y_pred_rf)
    sns.heatmap(cm_rf, annot=True, fmt=',d', cmap='Blues', ax=axes[0],
                xticklabels=['Benigno', 'DGA'], yticklabels=['Benigno', 'DGA'])
    axes[0].set_title(f'Random Forest (Acc: {acc_rf*100:.2f}%)')
    axes[0].set_xlabel('Predicción')
    axes[0].set_ylabel('Real')

    cm_xgb = confusion_matrix(y_test, y_pred_xgb)
    sns.heatmap(cm_xgb, annot=True, fmt=',d', cmap='Greens', ax=axes[1],
                xticklabels=['Benigno', 'DGA'], yticklabels=['Benigno', 'DGA'])
    axes[1].set_title(f'XGBoost (Acc: {acc_xgb*100:.2f}%)')
    axes[1].set_xlabel('Predicción')
    axes[1].set_ylabel('Real')

    plt.tight_layout()
    plt.savefig('matriz_confusion_comparativa.png', dpi=300)
    print("[+] Gráfica de Matrices de Confusión guardada en: 'matriz_confusion_comparativa.png'")

    # --------------------------------------------------------------------------
    # 6. IMPORTANCIA DE LAS CARACTERÍSTICAS (FEATURE IMPORTANCE)
    # --------------------------------------------------------------------------
    print("\n" + "="*65)
    print(" IMPORTANCIA DE CARACTERÍSTICAS (XGBOOST)")
    print("="*65)

    df_importance = pd.DataFrame({
        'Feature': features,
        'Importance': xgb_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    for idx, row in df_importance.iterrows():
        print(f" - {row['Feature']:<30}: {row['Importance']*100:.2f}%")

    # --------------------------------------------------------------------------
    # 7. GUARDAR LOS MODELOS ENTRENADOS
    # --------------------------------------------------------------------------
    print("\n[*] Guardando los modelos entrenados en disco...")
    joblib.dump(rf_model, MODEL_RF_FILE)
    joblib.dump(xgb_model, MODEL_XGB_FILE)
    print(f"[+] Modelo RF guardado en:  {os.path.abspath(MODEL_RF_FILE)}")
    print(f"[+] Modelo XGB guardado en: {os.path.abspath(MODEL_XGB_FILE)}")
    print("\n[+] ¡Fase de entrenamiento finalizada con éxito!")

if __name__ == "__main__":
    main()