import os
import math
import re
from collections import Counter
import joblib
import pandas as pd

# Archivos de entrada y salida
MODEL_FILE = "modelo_xgboost.joblib"
INPUT_DATASET = "dataset_dga_vs_benigno_final.csv"
OUTPUT_EVALUATION = "evaluacion_resultados_modelo.csv"
CHUNK_SIZE = 100000  # Procesa de 100,000 en 100,000 para optimizar memoria RAM

# ------------------------------------------------------------------------------
# 1. EXTRACCIÓN DE CARACTERÍSTICAS LÉXICAS
# ------------------------------------------------------------------------------
def shannon_entropy(s):
    if not s:
        return 0.0
    prob = [float(s.count(c)) / len(s) for c in dict(Counter(s))]
    return -sum([p * math.log(p, 2) for p in prob])

def max_consecutive_pattern(s, pattern):
    matches = re.findall(pattern, s)
    return max(len(m) for m in matches) if matches else 0

def extract_features_from_sld(sld):
    sld = str(sld).lower().strip()
    length = len(sld)

    if length == 0:
        return {
            'length': 0,
            'entropy': 0.0,
            'vowel_ratio': 0.0,
            'digit_ratio': 0.0,
            'consonant_ratio': 0.0,
            'special_ratio': 0.0,
            'max_consecutive_consonants': 0,
            'max_consecutive_digits': 0,
            'hex_ratio': 0.0
        }

    vowels = len(re.findall(r'[aeiou]', sld))
    digits = len(re.findall(r'[0-9]', sld))
    special = len(re.findall(r'[^a-z0-9]', sld))
    consonants = len(re.findall(r'[bcdfghjklmnpqrstvwxyz]', sld))
    hex_chars = len(re.findall(r'[0-9a-f]', sld))

    max_cons = max_consecutive_pattern(sld, r'[bcdfghjklmnpqrstvwxyz]+')
    max_digs = max_consecutive_pattern(sld, r'[0-9]+')

    return {
        'length': length,
        'entropy': round(shannon_entropy(sld), 4),
        'vowel_ratio': round(vowels / length, 4),
        'digit_ratio': round(digits / length, 4),
        'consonant_ratio': round(consonants / length, 4),
        'special_ratio': round(special / length, 4),
        'max_consecutive_consonants': max_cons,
        'max_consecutive_digits': max_digs,
        'hex_ratio': round(hex_chars / length, 4)
    }

# ------------------------------------------------------------------------------
# 2. PROCESAMIENTO, INFERENCIA Y EVALUACIÓN
# ------------------------------------------------------------------------------
def main():
    if not os.path.exists(MODEL_FILE):
        print(f"[ERROR] No se encuentra el archivo del modelo '{MODEL_FILE}'.")
        return
    if not os.path.exists(INPUT_DATASET):
        print(f"[ERROR] No se encuentra el archivo de datos '{INPUT_DATASET}'.")
        return

    print(f"[*] Cargando modelo entrenado desde '{MODEL_FILE}'...")
    model = joblib.load(MODEL_FILE)
    print("[+] Modelo cargado exitosamente.")

    print(f"[*] Procesando y evaluando '{INPUT_DATASET}' por bloques...")

    # Contadores globales
    total_tp = 0  # Verdaderos Positivos (Real 1, Pred 1)
    total_tn = 0  # Verdaderos Negativos (Real 0, Pred 0)
    total_fp = 0  # Falsos Positivos (Real 0, Pred 1)
    total_fn = 0  # Falsos Negativos (Real 1, Pred 0)

    first_chunk = True
    total_procesados = 0

    for chunk in pd.read_csv(INPUT_DATASET, chunksize=CHUNK_SIZE):
        chunk = chunk.dropna(subset=['sld', 'label']).copy()
        chunk['label'] = chunk['label'].astype(int)

        # 1. Extraer características
        features_list = [extract_features_from_sld(s) for s in chunk['sld']]
        df_features = pd.DataFrame(features_list)

        # 2. Inferencia
        probs_dga = model.predict_proba(df_features)[:, 1]
        preds_binarias = (probs_dga >= 0.5).astype(int)

        # 3. Asignar columnas solicitadas
        chunk['probabilidad_dga'] = probs_dga.round(4)
        chunk['clasificacion_modelo'] = ['MALICIOSO' if p == 1 else 'BENIGNO' for p in preds_binarias]

        # Etiqueta de diagnóstico comparativo contra la columna 'label'
        def clasificar_diagnostico(row, pred):
            real = row['label']
            if real == 1 and pred == 1:
                return 'TP (Verdadero Positivo)'
            elif real == 0 and pred == 0:
                return 'TN (Verdadero Negativo)'
            elif real == 0 and pred == 1:
                return 'FP (Falso Positivo)'
            else:
                return 'FN (Falso Negativo)'

        diagnosticos = [clasificar_diagnostico(r, p) for (_, r), p in zip(chunk.iterrows(), preds_binarias)]
        chunk['resultado_evaluacion'] = diagnosticos

        # 4. Actualizar contadores
        y_real = chunk['label'].values
        total_tp += int(((y_real == 1) & (preds_binarias == 1)).sum())
        total_tn += int(((y_real == 0) & (preds_binarias == 0)).sum())
        total_fp += int(((y_real == 0) & (preds_binarias == 1)).sum())
        total_fn += int(((y_real == 1) & (preds_binarias == 0)).sum())

        # 5. Guardar archivo por partes
        if first_chunk:
            chunk.to_csv(OUTPUT_EVALUATION, index=False, mode='w')
            first_chunk = False
        else:
            chunk.to_csv(OUTPUT_EVALUATION, index=False, mode='a', header=False)

        total_procesados += len(chunk)
        print(f"    [+] Evaluados {total_procesados:,} registros...", end="\r", flush=True)

    print(f"\n[+] Archivo de resultados guardado en: {os.path.abspath(OUTPUT_EVALUATION)}")

    # --------------------------------------------------------------------------
    # 3. CÁLCULO DE TASAS Y REPORTE ESTADÍSTICO
    # --------------------------------------------------------------------------
    total_reales_positivos = total_tp + total_fn
    total_reales_negativos = total_tn + total_fp

    fpr = (total_fp / total_reales_negativos * 100) if total_reales_negativos > 0 else 0.0
    fnr = (total_fn / total_reales_positivos * 100) if total_reales_positivos > 0 else 0.0
    precision = (total_tp / (total_tp + total_fp) * 100) if (total_tp + total_fp) > 0 else 0.0
    recall = (total_tp / total_reales_positivos * 100) if total_reales_positivos > 0 else 0.0
    accuracy = ((total_tp + total_tn) / total_procesados * 100) if total_procesados > 0 else 0.0

    print("\n" + "="*70)
    print("         REPORTE ESTADÍSTICO DE EVALUACIÓN (TASAS FP / FN)")
    print("="*70)
    print(f" Total de registros analizados:       {total_procesados:,}")
    print(f" - Clase 0 (Benignos Reales):         {total_reales_negativos:,}")
    print(f" - Clase 1 (DGAs Reales):             {total_reales_positivos:,}")
    print("-" * 70)
    print(f" [TP] Verdaderos Positivos (DGA detectado):        {total_tp:>9,}")
    print(f" [TN] Verdaderos Negativos (Benigno permitido):    {total_tn:>9,}")
    print(f" [FP] FALSOS POSITIVOS (Benigno bloqueado):        {total_fp:>9,}")
    print(f" [FN] FALSOS NEGATIVOS (DGA no detectado):         {total_fn:>9,}")
    print("-" * 70)
    print(f" Tasa de Falsos Positivos (FPR):       {fpr:.4f}%  (FP / Total Benignos)")
    print(f" Tasa de Falsos Negativos (FNR):       {fnr:.4f}%  (FN / Total DGAs)")
    print(f" Exactitud Global (Accuracy):          {accuracy:.4f}%")
    print(f" Precisión (Precision):                {precision:.4f}%")
    print(f" Sensibilidad (Recall / TPR):          {recall:.4f}%")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()