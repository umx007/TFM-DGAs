import os
import time
from playwright.sync_api import sync_playwright

BASE_URL = "https://dgarchive.caad.fkie.fraunhofer.de/site/families.html"
LOGIN_URL = "https://dgarchive.caad.fkie.fraunhofer.de/site/"
OUTPUT_FILE = "urls_dgas.txt"

# Ingresa tus credenciales aquí
HTTP_USER = "juan_jose_castro_ceballos"
HTTP_PASS = "xxxxxxxxxxxxxx"

def main():
    with sync_playwright() as p:
        print("[*] Iniciando navegador Chromium...")
        browser = p.chromium.launch(headless=False) 
        
        # Pasar credenciales HTTP al contexto del navegador
        context = browser.new_context(
            viewport={'width': 1280, 'height': 800},
            http_credentials={'username': HTTP_USER, 'password': HTTP_PASS}
        )
        page = context.new_page()

        # 1. Autenticación inicial navegando a la página de familias
        print(f"[*] Autenticando en: {LOGIN_URL}")
        try:
            page.goto(LOGIN_URL, timeout=60000)
            print("[+] Autenticación HTTP exitosa.")
        except Exception as e:
            print(f"[!] Nota sobre la navegación inicial: {e}")

        # 2. Navegar a la lista principal de DGAs
        print(f"[*] Navegando a la tabla principal: {BASE_URL}")
        page.goto(BASE_URL, wait_until="domcontentloaded")

        # 3. Extraer los enlaces de la tabla
        print("[*] Extraendo URLs de la lista...")
        links = page.locator("table tr td a").all()
        
        dga_urls = []
        for link in links:
            href = link.get_attribute("href")
            text = link.inner_text().strip()
            if href and text and not href.startswith("http"):
                # Convertir enlace relativo a URL absoluta
                full_url = page.evaluate("href => new URL(href, document.baseURI).href", href)
                dga_urls.append((text, full_url))

        total = len(dga_urls)
        print(f"[+] Se encontraron {total} referencias en total.\n")

        # 4. Guardar en el archivo de texto
        print(f"[*] Guardando URLs en '{OUTPUT_FILE}'...")
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(f"# Lista de URLs extraídas de DGArchive ({total} total)\n\n")
            for name, url in dga_urls:
                f.write(f"{name}: {url}\n")

        print(f"[+] Proceso concluido, archivo guardado exitosamente en: {os.path.abspath(OUTPUT_FILE)}")
        browser.close()

if __name__ == "__main__":
    main()