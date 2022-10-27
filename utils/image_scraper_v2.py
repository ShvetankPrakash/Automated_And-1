import selenium
from selenium import webdriver
import time
import requests
import os
from PIL import Image
import io
import hashlib
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC


DRIVER_PATH = '~/Desktop/chromedriver'

seen_more = False
# My addition to GitHub Repo Code
def scroll_down(wd):
    # Get scroll height.
    last_height = wd.execute_script("return document.body.scrollHeight")
    while True:
        # Scroll down to the bottom.
        wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        # Wait to load the page.
        time.sleep(2)
        # Calculate new scroll height and compare with last scroll height.
        new_height = wd.execute_script("return document.body.scrollHeight")
        global seen_more
        if new_height == last_height:
            if not seen_more:
                WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.XPATH, "//span[contains(., 'See more anyway')]"))).click()
            else:
                break
            seen_more = True

        last_height = new_height
 


def fetch_image_urls(query:str, max_links_to_fetch:int, wd:webdriver, sleep_between_interactions:int=1):
    
    def scroll_to_end(wd):
        wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(sleep_between_interactions)    

    # build the google query
    search_url = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img"

    # load the page
    wd.get(search_url.format(q=query))

    image_urls = set()
    image_count = 0
    results_start = 0

	# My addition to GitHub Repo Code
    scroll_down(wd) 
    wd.find_element(By.XPATH,"//input[@value='Show more results']").click()
    scroll_down(wd)

    while image_count < max_links_to_fetch:
        scroll_to_end(wd)

        # get all image thumbnail results
        thumbnail_results = wd.find_elements(By.CSS_SELECTOR, "img.Q4LuWd")
        number_results = len(thumbnail_results)
        
        print(f"Found: {number_results} search results. Extracting links from {results_start}:{number_results}")
        
        for img in thumbnail_results[results_start:number_results]:
            # try to click every thumbnail such that we can get the real image behind it
            try:
                img.click()
                time.sleep(sleep_between_interactions)
            except Exception:
                continue

            # extract image urls    
            actual_images = wd.find_elements(By.CSS_SELECTOR, 'img.n3VNCb')
            for actual_image in actual_images:
                if actual_image.get_attribute('src') and 'http' in actual_image.get_attribute('src'):
                    image_urls.add(actual_image.get_attribute('src'))

            image_count = len(image_urls)

            if len(image_urls) >= max_links_to_fetch:
                print(f"Found: {len(image_urls)} image links, done!")
                break
        else:
            print("Found:", len(image_urls), "image links, looking for more ...")
            time.sleep(30)
            return
            load_more_button = wd.find_element(By.CSS_SELECTOR, ".mye4qd")
            if load_more_button:
                wd.execute_script("document.querySelector('.mye4qd').click();")

        # move the result startpoint further down
        results_start = len(thumbnail_results)

    return image_urls


def persist_image(folder_path:str,file_name:str,url:str):
    try:
        image_content = requests.get(url, timeout=5).content
        #image_content.raise_for_status()

    except Exception as e:
        print(f"ERROR - Could not download {url} - {e}")

    try:
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file).convert('RGB')
        folder_path = os.path.join(folder_path,file_name)
        folder_path="/....../data/mask/" # absolute path needed
        if os.path.exists(folder_path):
            file_path = os.path.join(folder_path,hashlib.sha1(image_content).hexdigest()[:10] + '.jpg')
        else:
            os.mkdir(folder_path)
            file_path = os.path.join(folder_path,hashlib.sha1(image_content).hexdigest()[:10] + '.jpg')
        with open(file_path, 'wb') as f:
            image.save(f, "JPEG", quality=85)
        print(f"SUCCESS - saved {url} - as {file_path}")
    except Exception as e:
        print(f"ERROR - Could not save {url} - {e}")


if __name__ == '__main__':
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    driver.get("https://www.google.com")
    wd = driver
    queries = ["covid mask selfies"]
    for query in queries:
        wd.get('https://google.com')
        search_box = wd.find_element(By.CSS_SELECTOR, 'input.gLFyf')
        search_box.send_keys(query)
        links = fetch_image_urls(query, 250, wd)
        images_path = '/.../.../.../..../.../.../data'  # enter your desired image path (absolute path needed)
        for i in links:
            persist_image(images_path,query,i)
    wd.quit()
