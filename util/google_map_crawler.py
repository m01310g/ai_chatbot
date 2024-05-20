import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

def load_business_names_from_csv(csv_file_path, column_name):
    """
    CSV 파일에서 사업장명을 불러오는 함수
    :param csv_file_path: CSV 파일 경로
    :param column_name: 사업장명이 있는 열의 이름
    :return: 사업장명 리스트
    """
    df = pd.read_csv("E:/ai_chatbot/변형데이터/RestaurantData.csv")
    business_names = df['사업장명'].tolist()
    return business_names

def wait_input(driver):
    try:
        element = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.ID, "searchboxinput"))
        )
    except TimeoutException:
        print("검색창을 찾는 데 시간이 초과되었습니다.")
        driver.quit()
    return element

def input_function(query, driver):
    search_box = driver.find_element(By.ID, "searchboxinput")
    search_box.clear()
    search_box.send_keys(query)
    search_box.send_keys(Keys.ENTER)
    time.sleep(5)

def search_restaurants(business_names, driver):
    for name in business_names:
        query = name
        input_function(query, driver)
        for _ in range(5):  # 최대 5회까지 시도
            try:
                elements = driver.find_elements(By.CLASS_NAME, 'Nv2PK')
                if elements:
                    print(f"Found {len(elements)} elements")
                    for element in elements:
                        try:
                            print(element.text)
                            # 여기서 필요한 작업을 수행하면 됩니다. 예를 들어, 데이터를 저장하거나 출력할 수 있습니다.
                        except Exception as e:
                            print(f"Error appending food name: {e}")
                            break
                    break
                else:
                    print(f"No elements found for search term: {query}")
                    time.sleep(2)  # 잠시 대기 후 다시 시도
                    continue
            except NoSuchElementException:
                print(f"No elements found for search term: {query}")
                time.sleep(2)  # 잠시 대기 후 다시 시도
                continue
        else:
            print(f"No elements found after multiple attempts for search term: {query}")

def main():
    csv_file_path = 'E:/ai_chatbot/변형데이터/RestaurantData.csv'  # CSV 파일 경로
    business_names = load_business_names_from_csv(csv_file_path, '사업장명')

    options = webdriver.ChromeOptions()
    options.add_argument("--lang=ko")
    options.add_argument('disable-gpu')

    driver = webdriver.Chrome()
    link = 'https://www.google.com/maps'
    driver.get(link)

    wait_input(driver)

    search_restaurants(business_names, driver)

    driver.quit()

if __name__ == "__main__":
    main()
