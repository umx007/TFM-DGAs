import os
import time
import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

INPUT_FILE = "dataset_features_v3_ready.csv"
MODEL_V3_FILE = "modelo_xgboost_v3.joblib"


def main():
  if not os.path.exists(INPUT_FILE):
    print(f"[ERROR] No se encuentra '{INPUT_FILE}'.")
    return

  print(f"[*] Cargando dataset V3 desde '{INPUT_FILE}'...")
  df = pd.read_csv(INPUT_FILE)

  features = [c for c in df.columns if c not in ["sld", "label"]]
  X = df[features]
  y = df["label"]

  print(f"[*] Total de características evaluadas: {len(features)}")

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.20, random_state=42, stratify=y
  )

  print(f"[*] Entrenando XGBoost V3 con {len(X_train):,} muestras...")
  t0 = time.time()

  xgb_v3 = XGBClassifier(
      n_estimators=180,
      max_depth=12,
      learning_rate=0.1,
      subsample=0.8,
      colsample_bytree=0.8,
      n_jobs=-1,
      random_state=42,
      eval_metric="logloss",
  )
  xgb_v3.fit(X_train, y_train)
  print(f"[+] Entrenado en {time.time() - t0:.2f} segundos.")

  # Evaluación
  y_pred = xgb_v3.predict(X_test)
  y_proba = xgb_v3.predict_proba(X_test)[:, 1]

  auc = roc_auc_score(y_test, y_proba)
  print("\n" + "=" * 65)
  print(" REPORTE DE EVALUACIÓN MODELO v3 (WORDNINJA + TF-IDF)")
  print("=" * 65)
  print(
      classification_report(
          y_test, y_pred, target_names=["Benigno (0)", "DGA (1)"], digits=4
      )
  )
  print(f" ROC-AUC Score: {auc:.4f}")

  # Guardar
  joblib.dump(xgb_v3, MODEL_V3_FILE)
  print(f"\n[+] Modelo V3 guardado en '{MODEL_V3_FILE}'.")


if __name__ == "__main__":
  main()