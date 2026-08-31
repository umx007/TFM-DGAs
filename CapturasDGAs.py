import os
import time
from playwright.sync_api import sync_playwright

URLS_FILE = "urls_dgas.txt"
OUTPUT_DIR = "capturas_dgas"
LOGIN_URL = "https://dgarchive.caad.fkie.fraunhofer.de/site/families.html"
PAUSA_SEGUNDOS = 35

HTTP_USER = "juan_jose_castro_ceballos"
HTTP_PASS = "xxxxxxxxxxxx"

def parse_urls_file(filepath):
    dga_list = []
    if not os.path.exists(filepath):
        print(f"[ERROR] No se encontró el archivo '{filepath}'.")
        return dga_list

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                parts = line.split(":", 1)
                dga_name = parts[0].strip()
                url = parts[1].strip()
                if not url.startswith("http") and "http" in line:
                    url = line[line.index("http"):]
                dga_list.append((dga_name, url))
    return dga_list

def main():
    dga_items = parse_urls_file(URLS_FILE)
    if not dga_items:
        print("[X] No hay URLs para procesar. Abortando.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total = len(dga_items)
    print(f"[+] Se cargaron {total} URLs desde '{URLS_FILE}'.\n")

    with sync_playwright() as p:
        print("[*] Iniciando navegador Chromium...")
        browser = p.chromium.launch(headless=False) 
        
        context = browser.new_context(
            viewport={'width': 1280, 'height': 1020},
            http_credentials={'username': HTTP_USER, 'password': HTTP_PASS}
        )
        page = context.new_page()

        print(f"[*] Navegando a la página de inicio: {LOGIN_URL}")
        try:
            page.goto(LOGIN_URL, timeout=60000)
            print("[+] Autenticación exitosa.")
        except Exception as e:
            print(f"[!] Nota sobre la navegación inicial: {e}")

        # Sesión CDP para captura MHTML
        cdp = page.context.new_cdp_session(page)

        for index, (dga_name, url) in enumerate(dga_items, start=1):
            safe_name = dga_name.replace(":", "_").replace("/", "_").replace("\\", "_")
            png_path = os.path.join(OUTPUT_DIR, f"{safe_name}.png")
            mhtml_path = os.path.join(OUTPUT_DIR, f"{safe_name}.mhtml")

            print(f"[{index}/{total}] Procesando: {dga_name} -> {url}")

            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
                
                # 1. CAPTURA PNG
                target_element = page.locator("main")
                if target_element.count() == 0:
                    target_element = page.locator("h1").locator("..")

                target_element.wait_for(state="visible", timeout=10000)
                target_element.screenshot(path=png_path)
                print(f"    [OK] Screenshot guardado: {png_path}")

                # 2. CAPTURA MHTML (Guardado binario correcto)
                mhtml_data = cdp.send("Page.captureSnapshot", {"format": "mhtml"})
                with open(mhtml_path, "wb") as f:
                    f.write(mhtml_data["data"].encode("utf-8"))
                print(f"    [OK] Archivo MHTML válido guardado: {mhtml_path}")

            except Exception as e:
                print(f"    [ERROR] Falló el procesamiento para '{dga_name}': {e}")

            if index < total:
                print(f"    [*] Esperando {PAUSA_SEGUNDOS} segundos...")
                time.sleep(PAUSA_SEGUNDOS)

        print(f"\n[+] Proceso finalizado. Archivos guardados correctamente en '{OUTPUT_DIR}'.")
        browser.close()

if __name__ == "__main__":
    main()