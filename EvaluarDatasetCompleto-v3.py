import os
import math
import re
from collections import Counter
import joblib
import pandas as pd
import wordninja
from sklearn.feature_extraction.text import TfidfVectorizer

# Archivos de entrada y salida
MODEL_FILE = "modelo_xgboost_v3.joblib"
TFIDF_FILE = "tfidf_char_vectorizer.joblib"
INPUT_DATASET = "dataset_dga_vs_benigno_final.csv"
OUTPUT_EVALUATION = "evaluacion_resultados_modelo_v3.csv"
CHUNK_SIZE = 100000

# Diccionarios de referencia
COMMON_BIGRAMS = {
    'th': 0.0388, 'he': 0.0368, 'in': 0.0228, 'er': 0.0217, 'an': 0.0214, 're': 0.0175,
    'nd': 0.0157, 'on': 0.0141, 'en': 0.0138, 'at': 0.0133, 'ou': 0.0128, 'ed': 0.0127,
    'ha': 0.0127, 'to': 0.0117, 'or': 0.0115, 'it': 0.0113, 'is': 0.0111, 'hi': 0.0109,
    'es': 0.0109, 'ng': 0.0105, 'co': 0.0100, 'de': 0.0095, 'al': 0.0090, 'te': 0.0085,
    'se': 0.0080, 'la': 0.0075, 'me': 0.0070, 'ro': 0.0065
}

COMMON_TRIGRAMS = {
    'the': 0.0181, 'and': 0.0073, 'ing': 0.0072, 'her': 0.0036, 'hat': 0.0035, 'his': 0.0035,
    'tha': 0.0033, 'ere': 0.0031, 'for': 0.0028, 'ent': 0.0027, 'ion': 0.0026, 'ter': 0.0026,
    'was': 0.0025, 'you': 0.0025, 'ith': 0.0024, 'ver': 0.0024, 'all': 0.0023, 'wit': 0.0023,
    'thi': 0.0023, 'com': 0.0022, 'app': 0.0020, 'net': 0.0019, 'ser': 0.0018
}

def shannon_entropy(s):
    if not s:
        return 0.0
    prob = [float(s.count(c)) / len(s) for c in dict(Counter(s))]
    return -sum([p * math.log(p, 2) for p in prob])

def max_consecutive_pattern(s, pattern):
    matches = re.findall(pattern, s)
    return max(len(m) for m in matches) if matches else 0

def calculate_ngram_scores(sld):
    clean_s = re.sub(r'[^a-z]', '', sld)
    n = len(clean_s)
    if n < 2:
        return 0.0, 0.0
    bigrams = [clean_s[i:i+2] for i in range(n - 1)]
    bigram_score = sum(COMMON_BIGRAMS.get(bg, 0.0001) for bg in bigrams) / len(bigrams)
    if n < 3:
        trigram_score = 0.0
    else:
        trigrams = [clean_s[i:i+3] for i in range(n - 2)]
        trigram_score = sum(COMMON_TRIGRAMS.get(tg, 0.00005) for tg in trigrams) / len(trigrams)
    return round(bigram_score, 6), round(trigram_score, 6)

def vowel_consonant_transitions(sld):
    clean_s = re.sub(r'[^a-z]', '', sld)
    if len(clean_s) < 2:
        return 0.0
    vowels = set('aeiou')
    transitions = sum(1 for i in range(len(clean_s) - 1) if (clean_s[i] in vowels) != (clean_s[i+1] in vowels))
    return round(transitions / (len(clean_s) - 1), 4)

def extract_wordninja_features(sld):
    clean_s = re.sub(r'[^a-z]', '', sld)
    if not clean_s:
        return {'word_count': 0, 'avg_word_length': 0.0, 'valid_word_ratio': 0.0}
    
    words = wordninja.split(clean_s)
    word_count = len(words)
    if word_count == 0:
        return {'word_count': 0, 'avg_word_length': 0.0, 'valid_word_ratio': 0.0}
    
    avg_word_len = sum(len(w) for w in words) / word_count
    valid_words_chars = sum(len(w) for w in words if len(w) >= 3)
    valid_word_ratio = valid_words_chars / len(clean_s)
    
    return {
        'word_count': word_count,
        'avg_word_length': round(avg_word_len, 4),
        'valid_word_ratio': round(valid_word_ratio, 4)
    }

def extract_base_features(sld):
    sld = str(sld).lower().strip()
    length = len(sld)

    if length == 0:
        return {
            'length': 0, 'entropy': 0.0, 'vowel_ratio': 0.0, 'digit_ratio': 0.0,
            'consonant_ratio': 0.0, 'special_ratio': 0.0, 'max_consecutive_consonants': 0,
            'max_consecutive_digits': 0, 'hex_ratio': 0.0, 'bigram_score': 0.0,
            'trigram_score': 0.0, 'vc_transition_rate': 0.0,
            'word_count': 0, 'avg_word_length': 0.0, 'valid_word_ratio': 0.0
        }

    vowels = len(re.findall(r'[aeiou]', sld))
    digits = len(re.findall(r'[0-9]', sld))
    special = len(re.findall(r'[^a-z0-9]', sld))
    consonants = len(re.findall(r'[bcdfghjklmnpqrstvwxyz]', sld))
    hex_chars = len(re.findall(r'[0-9a-f]', sld))

    max_cons = max_consecutive_pattern(sld, r'[bcdfghjklmnpqrstvwxyz]+')
    max_digs = max_consecutive_pattern(sld, r'[0-9]+')
    bigram_score, trigram_score = calculate_ngram_scores(sld)
    vc_rate = vowel_consonant_transitions(sld)
    wn_features = extract_wordninja_features(sld)

    features = {
        'length': length,
        'entropy': round(shannon_entropy(sld), 4),
        'vowel_ratio': round(vowels / length, 4),
        'digit_ratio': round(digits / length, 4),
        'consonant_ratio': round(consonants / length, 4),
        'special_ratio': round(special / length, 4),
        'max_consecutive_consonants': max_cons,
        'max_consecutive_digits': max_digs,
        'hex_ratio': round(hex_chars / length, 4),
        'bigram_score': bigram_score,
        'trigram_score': trigram_score,
        'vc_transition_rate': vc_rate
    }
    features.update(wn_features)
    return features

def main():
    if not os.path.exists(MODEL_FILE):
        print(f"[ERROR] No se encuentra '{MODEL_FILE}'.")
        return
    if not os.path.exists(TFIDF_FILE):
        print(f"[ERROR] No se encuentra '{TFIDF_FILE}'.")
        return
    if not os.path.exists(INPUT_DATASET):
        print(f"[ERROR] No se encuentra '{INPUT_DATASET}'.")
        return

    print(f"[*] Cargando Modelo V3 ('{MODEL_FILE}') y Vectorizador TF-IDF ('{TFIDF_FILE}')...")
    model = joblib.load(MODEL_FILE)
    tfidf_vec = joblib.load(TFIDF_FILE)
    tfidf_feature_names = [f"tfidf_{name}" for name in tfidf_vec.get_feature_names_out()]
    print("[+] Componentes cargados exitosamente.")

    print(f"[*] Evaluando '{INPUT_DATASET}' por bloques...")

    total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0
    first_chunk = True
    total_procesados = 0

    for chunk in pd.read_csv(INPUT_DATASET, chunksize=CHUNK_SIZE):
        chunk = chunk.dropna(subset=['sld', 'label']).copy()
        chunk['label'] = chunk['label'].astype(int)

        # 1. Características base + Wordninja
        base_feats = [extract_base_features(s) for s in chunk['sld']]
        df_base = pd.DataFrame(base_feats)

        # 2. TF-IDF
        tfidf_mat = tfidf_vec.transform(chunk['sld'].astype(str)).toarray()
        df_tfidf = pd.DataFrame(tfidf_mat, columns=tfidf_feature_names)

        # 3. Vector completo (65 variables)
        df_features = pd.concat([df_base, df_tfidf], axis=1)

        # 4. Inferencia
        probs_dga = model.predict_proba(df_features)[:, 1]
        preds_binarias = (probs_dga >= 0.5).astype(int)

        chunk['probabilidad_dga'] = probs_dga.round(4)
        chunk['clasificacion_modelo'] = ['MALICIOSO' if p == 1 else 'BENIGNO' for p in preds_binarias]

        def clasificar_diagnostico(real, pred):
            if real == 1 and pred == 1:
                return 'TP (Verdadero Positivo)'
            elif real == 0 and pred == 0:
                return 'TN (Verdadero Negativo)'
            elif real == 0 and pred == 1:
                return 'FP (Falso Positivo)'
            else:
                return 'FN (Falso Negativo)'

        chunk['resultado_evaluacion'] = [clasificar_diagnostico(r, p) for r, p in zip(chunk['label'], preds_binarias)]

        y_real = chunk['label'].values
        total_tp += int(((y_real == 1) & (preds_binarias == 1)).sum())
        total_tn += int(((y_real == 0) & (preds_binarias == 0)).sum())
        total_fp += int(((y_real == 0) & (preds_binarias == 1)).sum())
        total_fn += int(((y_real == 1) & (preds_binarias == 0)).sum())

        if first_chunk:
            chunk.to_csv(OUTPUT_EVALUATION, index=False, mode='w')
            first_chunk = False
        else:
            chunk.to_csv(OUTPUT_EVALUATION, index=False, mode='a', header=False)

        total_procesados += len(chunk)
        print(f"    [+] Evaluados {total_procesados:,} registros...", end="\r", flush=True)

    print(f"\n[+] Guardado archivo de salida en: {os.path.abspath(OUTPUT_EVALUATION)}")

    total_reales_positivos = total_tp + total_fn
    total_reales_negativos = total_tn + total_fp

    fpr = (total_fp / total_reales_negativos * 100) if total_reales_negativos > 0 else 0.0
    fnr = (total_fn / total_reales_positivos * 100) if total_reales_positivos > 0 else 0.0
    precision = (total_tp / (total_tp + total_fp) * 100) if (total_tp + total_fp) > 0 else 0.0
    recall = (total_tp / total_reales_positivos * 100) if total_reales_positivos > 0 else 0.0
    accuracy = ((total_tp + total_tn) / total_procesados * 100) if total_procesados > 0 else 0.0

    print("\n" + "="*70)
    print("      REPORTE ESTADÍSTICO DE EVALUACIÓN V3 (WORDNINJA + TF-IDF)")
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