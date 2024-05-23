import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time

# CSV 파일 읽기
df = pd.read_csv('/변형데이터/CafeData.csv')

# 별점 데이터를 저장할 리스트 생성
ratings = []

# Selenium 설정
options = webdriver.ChromeOptions()
# options.add_argument('--headless')  # 브라우저를 숨기고 실행 (옵션)
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

for restaurant in df['사업장명']:
    # 구글 맵 검색 URL
    search_url = f"https://www.google.com/maps/search/{restaurant}"

    # 구글 맵 페이지 열기
    driver.get(search_url)
    time.sleep(3)  # 페이지 로딩 대기

    try:
        # 별점 요소 찾기
        rating_element = driver.find_element(By.CLASS_NAME, 'MW4etd')
        rating = rating_element.text
    except:
        rating = 'N/A'  # 별점을 찾을 수 없으면 'N/A'로 설정

    ratings.append(rating)
    print(f"{restaurant}: {rating}")

driver.quit()

# 별점 데이터를 데이터프레임에 추가
df['별점'] = ratings

# CSV 파일 저장
df.to_csv('E:/ai_chatbot/변형데이터/RestaurantData.csv', index=False)
