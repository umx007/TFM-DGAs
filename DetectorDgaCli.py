import os
import math
import re
import joblib
import pandas as pd
import tldextract
from collections import Counter

MODEL_FILE = "modelo_xgboost.joblib"

# --- 1. EXTRACTOR DE CARACTERÍSTICAS LÉXICAS AL VUELO ---
def shannon_entropy(s):
    if not s:
        return 0.0
    prob = [float(s.count(c)) / len(s) for c in dict(Counter(s))]
    return -sum([p * math.log(p, 2) for p in prob])

def max_consecutive_pattern(s, pattern):
    matches = re.findall(pattern, s)
    if not matches:
        return 0
    return max(len(m) for m in matches)

def extract_features(domain_input):
    """Extrae el SLD y calcula el vector de características."""
    extracted = tldextract.extract(str(domain_input))
    sld = extracted.domain.lower()
    length = len(sld)

    if length == 0:
        return None, None

    vowels = len(re.findall(r'[aeiou]', sld))
    digits = len(re.findall(r'[0-9]', sld))
    special = len(re.findall(r'[^a-z0-9]', sld))
    consonants = len(re.findall(r'[bcdfghjklmnpqrstvwxyz]', sld))
    hex_chars = len(re.findall(r'[0-9a-f]', sld))

    max_cons = max_consecutive_pattern(sld, r'[bcdfghjklmnpqrstvwxyz]+')
    max_digs = max_consecutive_pattern(sld, r'[0-9]+')

    features_dict = {
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
    return sld, pd.DataFrame([features_dict])

def main():
    if not os.path.exists(MODEL_FILE):
        print(f"[ERROR] No se encontró el modelo entrenado '{MODEL_FILE}'.")
        return

    print(f"[*] Cargando modelo '{MODEL_FILE}'...")
    model = joblib.load(MODEL_FILE)
    print("[+] Modelo listo para inferencia.\n")
    print("="*65)
    print("   DETECTOR AUTOMATIZADO DE DOMINIOS DGA (INFERENCIA EN VIVO)  ")
    print("   Escribe un dominio (ej: google.com, cmid1s1zeiu.life) o 'salir'")
    print("="*65 + "\n")

    while True:
        try:
            dominio_raw = input("Ingrese dominio a evaluar > ").strip()
            if not dominio_raw:
                continue
            if dominio_raw.lower() in ['salir', 'exit', 'quit']:
                print("\n[+] Cerrando detector.")
                break

            sld, df_vector = extract_features(dominio_raw)
            if df_vector is None:
                print("   [!] Entrada no válida.\n")
                continue

            # Inferencia
            prob_dga = model.predict_proba(df_vector)[0][1]
            prediccion = model.predict(df_vector)[0]

            print(f"   --> SLD Extraído: '{sld}'")
            print(f"   --> Entropía: {df_vector['entropy'].values[0]} | Longitud: {df_vector['length'].values[0]}")

            if prediccion == 1:
                print(f"   [ALERTA] Clasificación: DGA / MALICIOSO (Probabilidad: {prob_dga*100:.2f}%)\n")
            else:
                print(f"   [SEGURO] Clasificación: BENIGNO / LEGÍTIMO (Probabilidad DGA: {prob_dga*100:.2f}%)\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"   [!] Error durante la inferencia: {e}\n")

if __name__ == "__main__":
    main()