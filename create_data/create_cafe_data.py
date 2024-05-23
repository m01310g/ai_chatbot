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

cafe_df = pd.read_csv('C:/Users/hello/Downloads/07_24_05_P_CSV/fulldata_07_24_05_P_휴게음식점.csv', encoding='cp949')

cafe_df = cafe_df[['영업상태명', '사업장명', '소재지전체주소', '좌표정보(x)', '좌표정보(y)']]

# 필요없는 데이터(폐업 점포) 제거
cafe_df.drop(cafe_df[cafe_df['영업상태명'] == '폐업'].index, inplace=True)
cafe_df.drop(cafe_df[cafe_df['영업상태명'] == '취소/말소/만료/정지/중지'].index, inplace=True)

# 데이터 결측치 제거
cafe_df.dropna()

cafe_df['좌표정보(x)'] = pd.to_numeric(cafe_df['좌표정보(x)'], errors='coerce')
cafe_df['좌표정보(y)'] = pd.to_numeric(cafe_df['좌표정보(y)'], errors='coerce')
cafe_df.index = range(len(cafe_df))

# 좌표계 변환
coord = np.array(cafe_df[['좌표정보(x)', '좌표정보(y)']])
p1_type = 'epsg:2097'
p2_type = 'epsg:4326'
result = project_array(coord, p1_type, p2_type)

# 위경도로 변환한 값 데이터프레임에 추가
cafe_df['경도'] = result[:, 0]
cafe_df['위도'] = result[:, 1]

# 좌표정보 열 삭제
cafe_df.drop('좌표정보(x)', axis=1, inplace=True)
cafe_df.drop('좌표정보(y)', axis=1, inplace=True)