from pathlib import Path
import os
import re
import time
import requests
from urllib.parse import urljoin, urlparse

import shutil


from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

def clear_folder(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    for item in folder.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"]



def make_site_name(site_url):
    parsed = urlparse(site_url)
    domain = parsed.netloc.replace("www.", "")

    if not domain:
        domain = "site"

    site_name = re.sub(r"[^a-zA-Z0-9]+", "_", domain)
    return site_name.strip("_").lower()


def get_extension_from_url(img_url):
    ext = os.path.splitext(urlparse(img_url).path)[1].lower()

    if ext in IMAGE_EXTENSIONS:
        return ext

    return ".jpg"


def get_best_url_from_srcset(srcset):
    if not srcset:
        return None

    parts = srcset.split(",")
    last_part = parts[-1].strip()
    url = last_part.split(" ")[0].strip()

    return url


def clean_image_url(img_url):
    if not img_url:
        return None

    img_url = img_url.strip()

    if img_url.startswith("data:"):
        return None

    if img_url.startswith("blob:"):
        return None

    return img_url


def extract_image_urls_from_html(site_url, headers):
    image_urls = set()

    try:
        response = requests.get(site_url, headers=headers, timeout=20)
        response.raise_for_status()
    except Exception as error:
        print(f"Обычный HTML-парсер не смог открыть страницу: {error}")
        return image_urls

    soup = BeautifulSoup(response.text, "html.parser")

    for img in soup.find_all("img"):
        for attr in [
            "src",
            "data-src",
            "data-lazy-src",
            "data-original",
            "data-url",
            "data-image",
        ]:
            value = clean_image_url(img.get(attr))
            if value:
                image_urls.add(urljoin(site_url, value))

        best_url = get_best_url_from_srcset(img.get("srcset"))
        best_url = clean_image_url(best_url)
        if best_url:
            image_urls.add(urljoin(site_url, best_url))

    for source in soup.find_all("source"):
        best_url = get_best_url_from_srcset(source.get("srcset"))
        best_url = clean_image_url(best_url)
        if best_url:
            image_urls.add(urljoin(site_url, best_url))

    for meta in soup.find_all("meta"):
        property_value = meta.get("property") or meta.get("name")
        content = clean_image_url(meta.get("content"))

        if property_value in ["og:image", "twitter:image"] and content:
            image_urls.add(urljoin(site_url, content))

    return image_urls


def extract_image_urls_from_js(site_url, scroll_count=5):
    image_urls = set()

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)

            page = browser.new_page(
                viewport={"width": 1920, "height": 1080},
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            )

            print("Открываем страницу через браузер...")
            page.goto(site_url, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(3000)

            for _ in range(scroll_count):
                page.mouse.wheel(0, 3000)
                page.wait_for_timeout(1000)

            print("Собираем изображения после JS...")

            img_items = page.eval_on_selector_all(
                "img",
                """
                imgs => imgs.map(img => ({
                    src: img.getAttribute("src"),
                    realSrc: img.src,
                    currentSrc: img.currentSrc,
                    srcset: img.getAttribute("srcset"),
                    dataSrc: img.getAttribute("data-src"),
                    dataLazySrc: img.getAttribute("data-lazy-src"),
                    dataOriginal: img.getAttribute("data-original")
                }))
                """
            )

            for item in img_items:
                for key in ["src", "realSrc", "currentSrc", "dataSrc", "dataLazySrc", "dataOriginal"]:
                    value = clean_image_url(item.get(key))
                    if value:
                        image_urls.add(urljoin(site_url, value))

                best_url = get_best_url_from_srcset(item.get("srcset"))
                best_url = clean_image_url(best_url)
                if best_url:
                    image_urls.add(urljoin(site_url, best_url))

            source_srcsets = page.eval_on_selector_all(
                "source",
                """
                sources => sources.map(source => source.getAttribute("srcset"))
                """
            )

            for srcset in source_srcsets:
                best_url = get_best_url_from_srcset(srcset)
                best_url = clean_image_url(best_url)
                if best_url:
                    image_urls.add(urljoin(site_url, best_url))

            meta_images = page.eval_on_selector_all(
                "meta",
                """
                metas => metas
                    .filter(meta => {
                        const property = meta.getAttribute("property");
                        const name = meta.getAttribute("name");
                        return property === "og:image" || name === "twitter:image";
                    })
                    .map(meta => meta.getAttribute("content"))
                """
            )

            for img_url in meta_images:
                img_url = clean_image_url(img_url)
                if img_url:
                    image_urls.add(urljoin(site_url, img_url))

            browser.close()

    except Exception as error:
        print(f"JS-парсер не смог обработать страницу: {error}")

    return image_urls


def download_images_combined(site_url, output_dir=RAW_DIR):
    output_dir = Path(output_dir)
    clear_folder(output_dir)

    site_name = make_site_name(site_url)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,ru;q=0.8",
    }

    print("Пробуем обычный HTML-парсер...")
    html_urls = extract_image_urls_from_html(site_url, headers)
    print(f"HTML-парсер нашёл ссылок: {len(html_urls)}")

    print("Пробуем JS-парсер...")
    js_urls = extract_image_urls_from_js(site_url)
    print(f"JS-парсер нашёл ссылок: {len(js_urls)}")

    image_urls = list(html_urls | js_urls)
    print(f"Всего уникальных ссылок на изображения: {len(image_urls)}")

    saved_files = []

    for i, img_url in enumerate(image_urls):
        try:
            time.sleep(0.3)

            response = requests.get(img_url, headers=headers, timeout=20)
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")
            if "image" not in content_type:
                continue

            ext = get_extension_from_url(img_url)

            filename = f"{site_name}_combined_image_{i}{ext}"
            filepath = output_dir / filename

            counter = 1
            while filepath.exists():
                filename = f"{site_name}_combined_image_{i}_{counter}{ext}"
                filepath = output_dir / filename
                counter += 1

            with open(filepath, "wb") as file:
                file.write(response.content)

            saved_files.append(str(filepath))
            print(f"Скачано: {filepath}")

        except Exception as error:
            print(f"Не удалось скачать {img_url}: {error}")

    print(f"Всего скачано изображений: {len(saved_files)}")
    return saved_files


if __name__ == "__main__":
    url = input("Введите URL сайта: ")
    download_images_combined(url)