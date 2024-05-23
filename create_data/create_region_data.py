import pandas as pd
import numpy as np

region_data1 = pd.read_csv('E:/ai_data/metro_travel_log/Training/labeling_data/TL_csv/tc_sgg_시군구코드.csv')

region_data1 = region_data1[['SIDO_NM', 'SGG_NM', 'DONG_NM']]

region_data1 = region_data1.dropna()

# SIDO_NM에서 '시'를 삭제
region_data1['SIDO_NM'] = region_data1['SIDO_NM'].apply(lambda x: x[:-1] if x.endswith('시') else x)

# SGG_NM에서 '시', '군', '구'를 삭제
region_data1['SGG_NM'] = region_data1['SGG_NM'].apply(lambda x: x[:-1] if x.endswith(('시', '군', '구')) else x)

# DONG_NM에서 '동'을 삭제
region_data1['DONG_NM'] = region_data1['DONG_NM'].apply(lambda x: x[:-1] if x.endswith('동') else x)

region_data2 = pd.read_csv('E:/ai_data/metro_travel_log/Training/labeling_data/TL_csv/tc_sgg_시군구코드.csv')

region_data2 = region_data2[['SIDO_NM', 'SGG_NM', 'DONG_NM']]

region_data2 = region_data2.dropna()

total = pd.concat([region_data1, region_data2], axis=0)

print(total)

total.to_csv('E:/ai_chatbot/변형데이터/지역명데이터.csv')