import os
import time
import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

INPUT_FILE = "dataset_features_v2_ready.csv"
MODEL_V2_FILE = "modelo_xgboost_v2.joblib"


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] No se encuentra '{INPUT_FILE}'.")
        return

    print(f"[*] Cargando dataset V2 desde '{INPUT_FILE}'...")
    df = pd.read_csv(INPUT_FILE)

    features = [c for c in df.columns if c not in ["sld", "label"]]
    X = df[features]
    y = df["label"]

    print(f"[*] Características evaluadas ({len(features)}): {features}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(
        f"[*] Entrenando XGBoost con {len(X_train):,} muestras de entrenamiento..."
    )
    t0 = time.time()

    # Configuramos estimadores más profundos para aprovechar los n-gramas
    xgb_v2 = XGBClassifier(
        n_estimators=150,
        max_depth=12,
        learning_rate=0.1,
        n_jobs=-1,
        random_state=42,
        eval_metric="logloss",
    )
    xgb_v2.fit(X_train, y_train)
    print(f"[+] Entrenado en {time.time() - t0:.2f} segundos.")

    # Evaluación
    y_pred = xgb_v2.predict(X_test)
    y_proba = xgb_v2.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    print("\n" + "=" * 65)
    print(" REPORTE DE EVALUACIÓN MODELO V2 (WORDINESS / N-GRAMAS)")
    print("=" * 65)
    print(
        classification_report(
            y_test, y_pred, target_names=["Benigno (0)", "DGA (1)"], digits=4
        )
    )
    print(f" ROC-AUC Score: {auc:.4f}")

    # Importancia de las nuevas características
    print("\n" + "=" * 65)
    print(" NUEVA IMPORTANCIA DE CARACTERÍSTICAS")
    print("=" * 65)
    df_imp = pd.DataFrame(
        {"Feature": features, "Importance": xgb_v2.feature_importances_}
    ).sort_values(by="Importance", ascending=False)
    for _, row in df_imp.iterrows():
        print(f" - {row['Feature']:<30}: {row['Importance']*100:.2f}%")

    joblib.dump(xgb_v2, MODEL_V2_FILE)
    print(f"\n[+] Modelo V2 exportado exitosamente a '{MODEL_V2_FILE}'")


if __name__ == "__main__":
    main()