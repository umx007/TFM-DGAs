import math
import os
import re
from collections import Counter
from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel
import tldextract
import wordninja

MODEL_FILE = "modelo_xgboost_v3.joblib"
TFIDF_FILE = "tfidf_char_vectorizer.joblib"

if not os.path.exists(MODEL_FILE) or not os.path.exists(TFIDF_FILE):
  raise FileNotFoundError("No se encontraron los archivos del modelo o TF-IDF.")

print("[*] Cargando modelo v3 y vectorizador TF-IDF en memoria...")
model = joblib.load(MODEL_FILE)
tfidf_vec = joblib.load(TFIDF_FILE)
tfidf_feature_names = [
    f"tfidf_{name}" for name in tfidf_vec.get_feature_names_out()
]
print("[+] Microservicio listo para inferencia en tiempo real.")

app = FastAPI(
    title="DGA Detection Microservice API (v3)",
    description=(
        "Microservicio de inferencia en tiempo real con XGBoost v3 (Wordninja +"
        " TF-IDF) para detección de DGAs."
    ),
    version="3.0.0",
)

COMMON_BIGRAMS = {
    "th": 0.0388,
    "he": 0.0368,
    "in": 0.0228,
    "er": 0.0217,
    "an": 0.0214,
    "re": 0.0175,
    "nd": 0.0157,
    "on": 0.0141,
    "en": 0.0138,
    "at": 0.0133,
    "ou": 0.0128,
    "ed": 0.0127,
    "ha": 0.0127,
    "to": 0.0117,
    "or": 0.0115,
    "it": 0.0113,
    "is": 0.0111,
    "hi": 0.0109,
    "es": 0.0109,
    "ng": 0.0105,
    "co": 0.0100,
    "de": 0.0095,
    "al": 0.0090,
    "te": 0.0085,
    "se": 0.0080,
    "la": 0.0075,
    "me": 0.0070,
    "ro": 0.0065,
}

COMMON_TRIGRAMS = {
    "the": 0.0181,
    "and": 0.0073,
    "ing": 0.0072,
    "her": 0.0036,
    "hat": 0.0035,
    "his": 0.0035,
    "tha": 0.0033,
    "ere": 0.0031,
    "for": 0.0028,
    "ent": 0.0027,
    "ion": 0.0026,
    "ter": 0.0026,
    "was": 0.0025,
    "you": 0.0025,
    "ith": 0.0024,
    "ver": 0.0024,
    "all": 0.0023,
    "wit": 0.0023,
    "thi": 0.0023,
    "com": 0.0022,
    "app": 0.0020,
    "net": 0.0019,
    "ser": 0.0018,
}


class DomainRequest(BaseModel):
  domain: str


class DetectionResponse(BaseModel):
  domain: str
  sld: str
  is_dga: bool
  dga_probability: float
  action_recommended: str
  word_breakdown: list
  entropy: float


def shannon_entropy(s):
  if not s:
    return 0.0
  prob = [float(s.count(c)) / len(s) for c in dict(Counter(s))]
  return -sum([p * math.log(p, 2) for p in prob])


def max_consecutive_pattern(s, pattern):
  matches = re.findall(pattern, s)
  return max(len(m) for m in matches) if matches else 0


def calculate_ngram_scores(sld):
  clean_s = re.sub(r"[^a-z]", "", sld)
  n = len(clean_s)
  if n < 2:
    return 0.0, 0.0
  bigrams = [clean_s[i : i + 2] for i in range(n - 1)]
  bigram_score = sum(COMMON_BIGRAMS.get(bg, 0.0001) for bg in bigrams) / len(
      bigrams
  )
  if n < 3:
    trigram_score = 0.0
  else:
    trigrams = [clean_s[i : i + 3] for i in range(n - 2)]
    trigram_score = sum(
        COMMON_TRIGRAMS.get(tg, 0.00005) for tg in trigrams
    ) / len(trigrams)
  return round(bigram_score, 6), round(trigram_score, 6)


def vowel_consonant_transitions(sld):
  clean_s = re.sub(r"[^a-z]", "", sld)
  if len(clean_s) < 2:
    return 0.0
  vowels = set("aeiou")
  transitions = sum(
      1
      for i in range(len(clean_s) - 1)
      if (clean_s[i] in vowels) != (clean_s[i + 1] in vowels)
  )
  return round(transitions / (len(clean_s) - 1), 4)


def extract_features_complete(domain_input):
  extracted = tldextract.extract(str(domain_input))
  sld = extracted.domain.lower().strip()
  length = len(sld)
  if length == 0:
    return None, None, []

  clean_s = re.sub(r"[^a-z]", "", sld)
  words = wordninja.split(clean_s) if clean_s else []
  word_count = len(words)
  avg_word_len = (sum(len(w) for w in words) / word_count) if word_count else 0
  valid_words_chars = sum(len(w) for w in words if len(w) >= 3)
  valid_word_ratio = (valid_words_chars / len(clean_s)) if clean_s else 0

  vowels = len(re.findall(r"[aeiou]", sld))
  digits = len(re.findall(r"[0-9]", sld))
  special = len(re.findall(r"[^a-z0-9]", sld))
  consonants = len(re.findall(r"[bcdfghjklmnpqrstvwxyz]", sld))
  hex_chars = len(re.findall(r"[0-9a-f]", sld))

  max_cons = max_consecutive_pattern(sld, r"[bcdfghjklmnpqrstvwxyz]+")
  max_digs = max_consecutive_pattern(sld, r"[0-9]+")
  bigram_score, trigram_score = calculate_ngram_scores(sld)
  vc_rate = vowel_consonant_transitions(sld)

  base_dict = {
      "length": length,
      "entropy": round(shannon_entropy(sld), 4),
      "vowel_ratio": round(vowels / length, 4),
      "digit_ratio": round(digits / length, 4),
      "consonant_ratio": round(consonants / length, 4),
      "special_ratio": round(special / length, 4),
      "max_consecutive_consonants": max_cons,
      "max_consecutive_digits": max_digs,
      "hex_ratio": round(hex_chars / length, 4),
      "bigram_score": bigram_score,
      "trigram_score": trigram_score,
      "vc_transition_rate": vc_rate,
      "word_count": word_count,
      "avg_word_length": round(avg_word_len, 4),
      "valid_word_ratio": round(valid_word_ratio, 4),
  }
  df_base = pd.DataFrame([base_dict])

  tfidf_mat = tfidf_vec.transform([sld]).toarray()
  df_tfidf = pd.DataFrame(tfidf_mat, columns=tfidf_feature_names)

  df_vector = pd.concat([df_base, df_tfidf], axis=1)
  return sld, df_vector, words


@app.get("/")
def root():
  return {
      "service": "DGA Detection Microservice V3",
      "accuracy": "93.05%",
      "auc_roc": "0.9752",
      "docs": "/docs",
  }


@app.post("/predict", response_model=DetectionResponse)
def predict_domain(request: DomainRequest):
  if not request.domain or not request.domain.strip():
    raise HTTPException(status_code=400, detail="El dominio no puede estar vacío.")

  sld, df_vector, words = extract_features_complete(request.domain)
  if df_vector is None:
    raise HTTPException(
        status_code=400, detail="No se pudo extraer un SLD válido."
    )

  prob = float(model.predict_proba(df_vector)[0][1])

  # Política de decisión multinivel calibrada
  if prob >= 0.85:
    action = "BLOQUEAR (DGA Confirmado / C2 Activo)"
    is_dga = True
  elif prob >= 0.50:
    action = "ALERTA (Sospechoso / Cuarentena SIEM)"
    is_dga = True
  else:
    action = "PERMITIR (Tráfico Benigno)"
    is_dga = False

  return {
      "domain": request.domain,
      "sld": sld,
      "is_dga": is_dga,
      "dga_probability": round(prob, 4),
      "action_recommended": action,
      "word_breakdown": words,
      "entropy": float(df_vector["entropy"].values[0]),
  }