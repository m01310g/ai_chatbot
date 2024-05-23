import pandas as pd
import pyproj
import numpy as np
import folium

# 좌표계 변환 함수 정의
def project_array(coord, p1_type, p2_type):
    """
    좌표계 변환 함수
    :param coord: x, y 좌표 정보가 담긴 numpy array
    :param p1_type: 입력 좌표계 정보 e.g) epsg:5179
    :param p2_type: 출력 좌표계 정보 e.g) epsg: 4326
    :return:
    """

    p1 = pyproj.Proj(init=p1_type)
    p2 = pyproj.Proj(init=p2_type)
    fx, fy = pyproj.transform(p1, p2, coord[:, 0], coord[:, 1])

    return np.dstack([fx, fy])[0]

# cafe_file = pd.read_csv('C:/Users/hello/Downloads/07_24_05_P_CSV/fulldata_07_24_05_P_휴게음식점.csv', encoding='cp949')
restaurant_file1 = pd.read_csv('C:/Users/hello/Downloads/07_24_04_P_CSV/fulldata_07_24_04_P_일반음식점.csv', encoding='cp949')
restaurant_file2 = pd.read_csv('C:/Users/hello/Downloads/07_24_01_P_CSV/fulldata_07_24_01_P_관광식당.csv', encoding='cp949')

restaurant_df = pd.concat([restaurant_file1, restaurant_file2], axis=0)

restaurant_df = restaurant_df[['영업상태명', '사업장명', '소재지전체주소', '좌표정보(x)', '좌표정보(y)']]

# 필요없는 데이터(폐업 점포) 제거
restaurant_df.drop(restaurant_df[restaurant_df['영업상태명'] == '폐업'].index, inplace=True)
restaurant_df.drop(restaurant_df[restaurant_df['영업상태명'] == '취소/말소/만료/정지/중지'].index, inplace=True)


# 데이터 결측치 제거
restaurant_df = restaurant_df.dropna()

restaurant_df['좌표정보(x)'] = pd.to_numeric(restaurant_df['좌표정보(x)'], errors='coerce')
restaurant_df['좌표정보(y)'] = pd.to_numeric(restaurant_df['좌표정보(y)'], errors='coerce')
restaurant_df.index = range(len(restaurant_df))

# 좌표계 변환
coord = np.array(restaurant_df[['좌표정보(x)', '좌표정보(y)']])
p1_type = 'epsg:2097'
p2_type = 'epsg:4326'
result = project_array(coord, p1_type, p2_type)

# 위경도로 변환한 값 데이터프레임에 추가
restaurant_df['경도'] = result[:, 0]
restaurant_df['위도'] = result[:, 1]

# 좌표정보 열 삭제
restaurant_df.drop('좌표정보(x)', axis=1, inplace=True)
restaurant_df.drop('좌표정보(y)', axis=1, inplace=True)

# 시도, 시군구, 읍면동 데이터 입력한 열 생성
restaurant_df['지역'] = restaurant_df['소재지전체주소'].apply(lambda x: x.split()[0:3])

region = restaurant_df['소재지전체주소'].apply(lambda x: x.split()[0:3])

region = region.tolist()

restaurant_df.to_csv('E:/ai_chatbot/변형데이터/RestaurantData.csv')

# for i in range(len(region)):
#     restaurant_df['시도'] = region[i][0]

# print(restaurant_df['지역'])
# for i in range(len(restaurant_df)):
#     restaurant_df['시군구'] = region[i][1]

# for i in range(len(region)):
#     restaurant_df['읍면동'] = region[i][2]