import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException, StaleElementReferenceException, ElementNotInteractableException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time


def load_business_names_from_csv(csv_file_path, column_name):
    df = pd.read_csv(csv_file_path)
    business_names = df[column_name].tolist()
    return business_names


def wait_input(driver):
    try:
        element = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.ID, "searchboxinput"))
        )
    except TimeoutException:
        print("검색창을 찾는 데 시간이 초과되었습니다.")
        driver.quit()
        return None
    return element


def input_function(query, driver):
    try:
        search_box = WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable((By.ID, "searchboxinput"))
        )
        search_box.clear()
        search_box.send_keys(query)
        search_box.send_keys(Keys.ENTER)
        time.sleep(5)  # 검색 결과가 로드될 때까지 대기 시간을 늘립니다.
    except (TimeoutException, ElementNotInteractableException) as e:
        print(f"Error interacting with the search box: {e}")
        return False
    return True


def extract_data(driver):
    try:
        # 첫 번째 검색 결과 클릭
        first_result = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, '.Nv2PK'))
        )
        first_result.click()
        time.sleep(5)  # 패널이 로드될 때까지 대기 시간을 늘립니다.

        try:
            name = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'h1 span:nth-child(1)'))
            ).text
        except NoSuchElementException:
            name = "N/A"

        try:
            rating = driver.find_element(By.CSS_SELECTOR, 'span[class*="MW4etd"]').text
        except NoSuchElementException:
            rating = "N/A"

        try:
            address = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, '[data-item-id="address"]'))
            ).get_attribute('aria-label')
        except NoSuchElementException:
            address = "N/A"

    except (TimeoutException, StaleElementReferenceException) as e:
        print(f"Error: {e}")
        name = rating = address = "N/A"

    return name, rating, address[4:] if address != "N/A" else address


def search_restaurants(business_names, driver):
    data = []
    seen = set()  # 중복 확인을 위한 집합
    for name in business_names:
        if not input_function(name, driver):
            continue  # Skip to next if input failed

        for _ in range(5):  # 최대 5회까지 시도
            try:
                # 검색 결과가 로드될 때까지 대기
                WebDriverWait(driver, 10).until(
                    EC.presence_of_all_elements_located((By.CLASS_NAME, 'Nv2PK'))
                )
                extracted_name, rating, address = extract_data(driver)
                if (extracted_name, rating, address) not in seen:
                    print(f"Original Name: {name}, Extracted Name: {extracted_name}, Rating: {rating}, Address: {address}")
                    data.append({"Original Name": name, "Extracted Name": extracted_name, "Rating": rating, "Address": address})
                    seen.add((extracted_name, rating, address))
                break
            except (NoSuchElementException, TimeoutException) as e:
                print(f"No elements found for search term: {name}, error: {e}")
                time.sleep(2)  # 잠시 대기 후 다시 시도
                continue
        else:
            print(f"No elements found after multiple attempts for search term: {name}")
    return data


def main():
    csv_file_path = '/변형데이터/RestaurantData.csv'  # CSV 파일 경로
    business_names = load_business_names_from_csv(csv_file_path, '사업장명')

    options = webdriver.ChromeOptions()
    options.add_argument("--lang=ko")
    options.add_argument('disable-gpu')

    driver = webdriver.Chrome(options=options)
    link = 'https://www.google.com/maps'
    driver.get(link)

    if wait_input(driver) is None:
        return  # Exit if search box is not found

    data = search_restaurants(business_names, driver)

    # DataFrame으로 변환 후 CSV 파일로 저장
    df = pd.DataFrame(data)
    df.to_csv('E:/ai_chatbot/리뷰데이터/RestaurantReviewData.csv', index=False, encoding='utf-8-sig')

    driver.quit()

if __name__ == "__main__":
    main()
