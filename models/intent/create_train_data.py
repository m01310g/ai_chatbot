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
nature = []
restaurant = []
cafe = []
# history = []
leisure = []
etc = []
# 의도 분류 데이터 필요시 추가 생성 예정

keywords_nature = ['국립공원', '도립공원', '군립공원', '산', '자연생태관광지', '자연휴양림', '수목원', '폭포', '계곡', '약수터', '해안절경', '해수욕장', '섬', '항구', '포구', '등대', '호수', '강', '동굴']
keywords_restaurant = ['밥', '음식점', '식당', '맛집', '음식', '식사', '아침', '점심', '저녁', '밥집', '식사']  # 식당 키워드 리스트
keywords_cafe = ['빵', '카페', '커피', '디저트', '음료', '케이크']  # 카페 키워드 리스트
# keywords_history = ['고궁', '성', '문', '고택', '생가', '민속마을', '유적지', '사적지', '사찰', '종교성지', '안보관광']
keywords_leisure = ['항공레포츠', '수련시설', '경기장', '인라인(실내 인라인 포함)', '자전거하이킹', '카트', '골프', '경마', '경륜', '카지노', '승마',
                   '스키', '스노보드', '스케이트', '썰매장', '수렵장', '사격장', '야영장', '오토캠핑장', '암벽등반', '서바이벌게임', 'ATV', 'MTB', '오프로드', '번지점프',
                   '스키(보드) 렌탈샵', '트래킹', '윈드서핑', '제트스키', '카약', '카누', '요트', '래프팅', '스카이다이빙', '초경량비행', '헹글라이딩', '패러글라이딩', '열기구', '복합 레포츠']

keywords_region = ['부산', '서울', '강원도', '제주', '대구', '대전', '광주', '울산', '인천', '경기도', '전라도', '충청도', '경상도']

# 지역명 제거
all_data = [text for text in all_data if not any(keyword in text for keyword in keywords_region)]

for i in all_data:
    if any(keyword in i for keyword in keywords_restaurant):
        restaurant.append(i)
    elif any(keyword in i for keyword in keywords_cafe):
        cafe.append(i)
    elif any(keyword in i for keyword in keywords_nature):
        nature.append(i)
    # elif any(keyword in i for keyword in keywords_history):
    #     history.append(i)
    elif any(keyword in i for keyword in keywords_leisure):
        leisure.append(i)
    else:
        etc.append(i)

restaurant_label = []
for _ in range(len(restaurant)):
    restaurant_label.append(0)

cafe_label = []
for _ in range(len(cafe)):
    cafe_label.append(1)

nature_label = []
for _ in range(len(nature)):
    nature_label.append(2)

# history_label = []
# for _ in range(len(history)):
#     history_label.append(3)

leisure_label = []
for _ in range(len(leisure)):
    leisure_label.append(3)

train_df = pd.DataFrame({'text' : restaurant+cafe+nature+leisure,
                         'label': restaurant_label+cafe_label+nature_label+leisure_label})


train_df.reset_index(drop=True, inplace=True)
train_df.to_csv('E:/ai_chatbot/models/intent/train_data.csv', index=False)