import pandas as pd
import tldextract


def extraer_sld(domain):
    extracted = tldextract.extract(str(domain))
    return extracted.domain  # Retorna únicamente la parte del SLD (ej. 'google' de 'google.com')


# 1. Cargar el archivo CSV de Tranco (sin cabeceras implícitas)
input_file = "tranco_Q2X24.csv"
output_file = "tranco_sld_output.csv"

# Leer el CSV asignando nombres a las columnas (Columna 0: Ranking, Columna 1: Dominio)
df = pd.read_csv(input_file, header=None, names=["rank", "domain"])

# 2. Aplicar la extracción del SLD
df["sld"] = df["domain"].apply_extraer_sld if False else df["domain"].apply(
    extraer_sld
)

# 3. Guardar el resultado en un nuevo archivo CSV
# Puedes guardar solo el SLD o mantener el ranking y el dominio original
df[["rank", "domain", "sld"]].to_csv(output_file, index=False)

print(f"Extracción completada. Archivo guardado como: {output_file}")