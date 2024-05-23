import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from konlpy.tag import Komoran

# 데이터 불러오기
purpose_data = pd.read_csv('E:/ai_chatbot/변형데이터/용도별목적대화데이터.csv')
topic_data = pd.read_csv('E:/ai_chatbot/변형데이터/주제별일상대화데이터.csv')
common_sense_data = pd.read_csv('E:/ai_chatbot/변형데이터/일반상식.csv')
region_data = pd.read_csv('E:/ai_chatbot/변형데이터/지역명데이터.csv')
add_data = pd.read_csv('E:/ai_chatbot/변형데이터/AllData.csv')\

# 결측값 제거
purpose_data.dropna(inplace=True)
topic_data.dropna(inplace=True)
common_sense_data.dropna(inplace=True)
region_data.dropna(inplace=True)
add_data.dropna(inplace=True)

all_data = list(purpose_data['text']) + list(topic_data['text']) + list(common_sense_data['query']) + list(common_sense_data['answer'])  + list(add_data['req']) + list(add_data['res']) + list(region_data['SIDO_NM']) + list(region_data['SGG_NM']) + list(region_data['DONG_NM'])

total = pd.DataFrame({'text': all_data})
total.to_csv('E:/ai_chatbot/변형데이터/통합본데이터.csv', index=False)

# 의도 분류 데이터 생성
restaurant = []
cafe = []
tour = []
travel = []
etc = []

keywords_restaurant = ['밥', '음식점', '식당', '맛집', '음식', '식사', '아침', '점심', '저녁', '밥집', '식사']  # 식당 키워드 리스트
keywords_cafe = ['빵', '카페', '커피', '디저트', '음료', '케이크']  # 카페 키워드 리스트
keywords_tour = ['관광', '장소', '명소', '여행지']  # 관광지 키워드 리스트
keywords_travel = ['여행', '계획', '플랜', '일정', '경로'] # 여행 키워드 리스트

for i in all_data:
    if any(keyword in i for keyword in keywords_restaurant):
        restaurant.append(i)
    elif any(keyword in i for keyword in keywords_cafe):
        cafe.append(i)
    elif any(keyword in i for keyword in keywords_tour):
        tour.append(i)
    elif any(keyword in i for keyword in keywords_travel):
        travel.append(i)
    else:
        etc.append(i)

restaurant_label = []
for _ in range(len(restaurant)):
    restaurant_label.append(0)

cafe_label = []
for _ in range(len(cafe)):
    cafe_label.append(1)

tour_label = []
for _ in range(len(tour)):
    tour_label.append(2)

travel_label = []
for _ in range(len(travel)):
    travel_label.append(3)

train_df = pd.DataFrame({'text' : restaurant+cafe+tour+travel,
                         'label': restaurant_label+cafe_label+tour_label+travel_label})


train_df.reset_index(drop=True, inplace=True)
train_df.to_csv('E:/ai_chatbot/models/intent/train_data.csv', index=False)