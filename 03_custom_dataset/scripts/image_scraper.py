import requests
from bs4 import BeautifulSoup
import os
import time
import re
import hashlib
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from PIL import Image
import io

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "download_dir_name": "BeeWPollen",
    "num_images_to_download": 100,
    "scroll_pause_time": 3,
    "user_agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    "headless": True,
    "file_prefix": "beewpollen6",
    "search_query": "bumblebee with pollen balls"
}


def calculate_hash(image_bytes):
    try:
        return hashlib.md5(image_bytes).hexdigest()
    except Exception as e:
        print(f"[ERROR] Failed to calculate hash: {e}")
        return None


def extract_image_urls(driver):
    try:
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')
        pattern = re.compile(r'\"(https:\/\/[^"]+\.(?:jpg|png|jpeg|webp))\"')
        matches = pattern.findall(str(soup))
        unique_urls = list(set(matches))
        cleaned_urls = [url.replace('\\', '') for url in unique_urls]
        print(f"[INFO] Extracted {len(cleaned_urls)} unique image URLs from current page")
        return cleaned_urls
    except Exception as e:
        print(f"[ERROR] Failed to extract image URLs: {e}")
        return []


def get_existing_images(download_dir):
    try:
        pattern = re.compile(rf'{CONFIG["file_prefix"]}_(\d+)\.jpg')
        existing_files = []

        if not os.path.exists(download_dir):
            print(f"[INFO] Download directory does not exist, will be created")
            return existing_files

        for filename in os.listdir(download_dir):
            match = pattern.match(filename)
            if match:
                number = int(match.group(1))
                existing_files.append((number, os.path.join(download_dir, filename)))

        print(f"[INFO] Found {len(existing_files)} existing images in download directory")
        return existing_files
    except Exception as e:
        print(f"[ERROR] Failed to get existing images: {e}")
        return []


def download_images(query):
    try:
        print("[INFO] Starting image download process")
        print(f"[INFO] Search query: '{query}'")
        print(f"[INFO] Target count: {CONFIG['num_images_to_download']} images")

        options = webdriver.ChromeOptions()
        if CONFIG["headless"]:
            options.add_argument("--headless=new")
            print("[INFO] Running Chrome in headless mode")
        options.add_argument(f"user-agent={CONFIG['user_agent']}")
        options.add_argument("--disable-blink-features=AutomationControlled")

        print("[INFO] Initializing web driver")
        driver = None
        try:
            driver = webdriver.Chrome(options=options)
            wait = WebDriverWait(driver, 20)
        except Exception as e:
            print(f"[ERROR] Failed to initialize web driver: {e}")
            return

        download_dir = os.path.join(CURRENT_DIR, CONFIG["download_dir_name"])
        os.makedirs(download_dir, exist_ok=True)
        print(f"[INFO] Target folder: {os.path.abspath(download_dir)}")

        existing_files = get_existing_images(download_dir)
        existing_hashes = set()

        print("[INFO] Calculating hashes of existing images")
        for _, filepath in existing_files:
            try:
                with open(filepath, 'rb') as f:
                    img_data = f.read()
                hash_value = calculate_hash(img_data)
                if hash_value:
                    existing_hashes.add(hash_value)
            except Exception as e:
                print(f"[WARNING] Could not process existing file {filepath}: {e}")

        if existing_files:
            max_number = max(number for number, _ in existing_files)
            start_index = max_number + 1
            print(f"[INFO] Starting file numbering from {start_index}")
        else:
            start_index = 1
            print("[INFO] No existing files, starting numbering from 1")

        processed_urls = set()
        downloaded_count = 0
        skipped_count = 0
        error_count = 0

        try:
            print(f"[INFO] Navigating to Google Images search page")
            driver.get(f"https://www.google.com/search?q={query}&tbm=isch")
            time.sleep(2)

            while downloaded_count < CONFIG["num_images_to_download"]:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                print(f"[INFO] Scrolling down for more images, waiting {CONFIG['scroll_pause_time']} seconds")
                time.sleep(CONFIG["scroll_pause_time"])

                try:
                    more_button = wait.until(
                        EC.element_to_be_clickable((By.XPATH, "//input[@value='Daha fazla sonuç göster']")))
                    print("[INFO] Clicking 'Show more results' button")
                    more_button.click()
                    time.sleep(2)
                except Exception:
                    pass

                image_urls = extract_image_urls(driver)
                new_images = [url for url in image_urls if url not in processed_urls]

                if not new_images:
                    print("[WARNING] No new images found")
                    break

                print(f"[INFO] Found {len(new_images)} new image URLs to process")

                for idx, img_url in enumerate(new_images):
                    if downloaded_count >= CONFIG["num_images_to_download"]:
                        break

                    processed_urls.add(img_url)

                    try:
                        print(
                            f"[PROGRESS] ({downloaded_count + 1}/{CONFIG['num_images_to_download']}) Downloading: {img_url[:60]}...")
                        response = requests.get(img_url, timeout=10)
                        response.raise_for_status()

                        img = Image.open(io.BytesIO(response.content))
                        if img.mode in ('RGBA', 'LA', 'P'):
                            img = img.convert('RGB')

                        buffer = io.BytesIO()
                        img.save(buffer, format='JPEG', quality=85)
                        jpeg_data = buffer.getvalue()
                        current_hash = calculate_hash(jpeg_data)

                        if current_hash not in existing_hashes:
                            image_number = start_index + downloaded_count
                            while True:
                                filename = f"{CONFIG['file_prefix']}_{image_number}.jpg"
                                filepath = os.path.join(download_dir, filename)
                                if not os.path.exists(filepath):
                                    break
                                image_number += 1

                            with open(filepath, 'wb') as f:
                                f.write(jpeg_data)

                            existing_hashes.add(current_hash)
                            downloaded_count += 1
                            print(
                                f"[INFO] Saved image {downloaded_count}/{CONFIG['num_images_to_download']} as {filename}")
                        else:
                            print(f"[WARNING] Duplicate image detected, skipping: {img_url[:60]}...")
                            skipped_count += 1

                    except Exception as e:
                        print(f"[ERROR] Failed to download image {img_url[:60]}...: {e}")
                        error_count += 1

        finally:
            if driver:
                driver.quit()
                print("[INFO] Web driver closed")

            print(f"[INFO] Download summary:")
            print(f"[INFO] - Successfully downloaded: {downloaded_count} images")
            print(f"[INFO] - Skipped duplicates: {skipped_count} images")
            print(f"[INFO] - Errors encountered: {error_count} images")
            print(f"[INFO] Image download process completed")

    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
        if driver:
            driver.quit()


if __name__ == "__main__":
    download_images(CONFIG["search_query"])