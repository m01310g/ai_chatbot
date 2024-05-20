import self
from konlpy.tag import Okt
import pandas as pd

from util.Preprocess import Preprocess
from models.intent.IntentModel import IntentModel

# Okt 형태소 분석기 인스턴스 생성
okt = Okt()

def extracted_region(user_query):
    # 형태소 분석
    morphs = okt.pos(user_query)

    location_keywords = ['시', '군', '구', '동', '면', '읍', '리']

    # 제외할 관련 키워드 리스트
    keywords = ['밥', '음식점', '식당', '맛집', '음식', '식사', '아침', '점심', '저녁', '밥집', '식사', '좀', '근처', '주변', '관광지', '관광', '명소', '카페', '커피', '케이크', '음료', '음료수']

    # 지역명 추출
    extracted_regions = []
    # temp_region = []

    for word, tag in morphs:
        if tag in ['Noun', 'ProperNoun'] and word not in keywords:  # 명사나 고유명사인 경우
            extracted_regions.append(word)

    # 후처리
    if extracted_regions:
        # 추출된 명사들 중 장소명으로 추정되는 부분을 연결
        for key in location_keywords:
            if key in extracted_regions:
                idx = extracted_regions.index(key)
                extracted_regions[idx-1] = (extracted_regions[idx-1] + extracted_regions[idx])
                del extracted_regions[idx]
        return extracted_regions
    else:
        return None

def get_restaurant_info(location):
    # 리스트가 전달되었다면 첫 번째 항목을 사용
    # location = location.split()
    if isinstance(location, list):
        if len(location) > 0:
            location = location[-1]
        else:
            return "지역 정보가 비어있습니다."
    elif location is None:
        return "지역을 찾을 수 없습니다."

    data = pd.read_csv('E:/ai_chatbot/변형데이터/RestaurantData.csv',
                       usecols=['사업장명', '소재지전체주소'])

    location_parts = location.split(' ')
    primary_location = location_parts[0]
    secondary_location = location_parts[1] if len(location_parts) > 1 else ''

    filtered_data_primary = data[data['소재지전체주소'].str.contains(primary_location, case=False, na=False)]

    if filtered_data_primary.empty:
        return "해당 지역의 식당 정보를 찾을 수 없습니다."

    if secondary_location:
        filtered_data_secondary = filtered_data_primary[
            filtered_data_primary['소재지전체주소'].str.contains(secondary_location, case=False, na=False)]
        if filtered_data_secondary.empty:
            return "해당 지역의 식당 정보를 찾을 수 없습니다."
        filtered_data = filtered_data_secondary
    else:
        filtered_data = filtered_data_primary

    restaurant_info = filtered_data['사업장명'].tolist()
    return restaurant_info

def get_cafe_info(location):
    # 리스트가 전달되었다면 첫 번째 항목을 사용
    # location = location.split()
    if isinstance(location, list):
        if len(location) > 0:
            location = location[-1]
        else:
            return "지역 정보가 비어있습니다."
    elif location is None:
        return "지역을 찾을 수 없습니다."

    data = pd.read_csv('E:/ai_chatbot/변형데이터/CafeData.csv',
                       usecols=['사업장명', '소재지전체주소'])

    location_parts = location.split(' ')
    primary_location = location_parts[0]
    secondary_location = location_parts[1] if len(location_parts) > 1 else ''

    filtered_data_primary = data[data['소재지전체주소'].str.contains(primary_location, case=False, na=False)]

    if filtered_data_primary.empty:
        return "해당 지역의 식당 정보를 찾을 수 없습니다."

    if secondary_location:
        filtered_data_secondary = filtered_data_primary[
            filtered_data_primary['소재지전체주소'].str.contains(secondary_location, case=False, na=False)]
        if filtered_data_secondary.empty:
            return "해당 지역의 식당 정보를 찾을 수 없습니다."
        filtered_data = filtered_data_secondary
    else:
        filtered_data = filtered_data_primary

    cafe_info = filtered_data['사업장명'].tolist()
    return cafe_info

def get_tour_info(location):
    if isinstance(location, list):
        if len(location) > 0:
            location = location[-1]
        else:
            return "지역 정보가 비어있습니다."
    elif location is None:
        return "지역을 찾을 수 없습니다."

    data = pd.read_csv('E:/ai_chatbot/변형데이터/전국관광지정보표준데이터.csv', encoding='cp949',
                       usecols=['소재지지번주소', '관광지명'])

    location_parts = location.split(' ')
    primary_location = location_parts[0]
    secondary_location = location_parts[1] if len(location_parts) > 1 else ''

    filtered_data_primary = data[data['소재지지번주소'].str.contains(primary_location, case=False, na=False)]

    if filtered_data_primary.empty:
        return "해당 지역의 관광지 정보를 찾을 수 없습니다."

    if secondary_location:
        filtered_data_secondary = filtered_data_primary[
            filtered_data_primary['소재지지번주소'].str.contains(secondary_location, case=False, na=False)]
        if filtered_data_secondary.empty:
            return "해당 지역의 관광지 정보를 찾을 수 없습니다."
        filtered_data = filtered_data_secondary
    else:
        filtered_data = filtered_data_primary

    tour_info = filtered_data['관광지명'].tolist()
    return tour_info


class Response:
    def __init__(self, user_intent, user_query):
        self.user_intent = user_intent
        self.user_query = user_query
    def generate_response(self, user_intent, user_query):
        location = extracted_region(user_query)
        if location is None:
            return '지역명을 찾을 수 없습니다.'

        if user_intent == '식당':
            restaurant_list = get_restaurant_info(location)
            location = ' '.join(location)

            # restaurant_list가 빈 문자열이 아니거나 특정 조건을 만족하는 경우
            # if restaurant_list and not restaurant_list.startswith('None'):
            if restaurant_list:
                response = f'{location}에 있는 음식점 정보\n{restaurant_list}'
            else:
                response = '해당 지역의 식당 정보를 검색할 수 없습니다.'

        elif user_intent == '카페':
            cafe_list = get_cafe_info(location)
            location = ' '.join(location)

            # cafe_list가 빈 문자열이 아니거나 특정 조건을 만족하는 경우
            # if cafe_list and not cafe_list.startswith('None'):
            if cafe_list:
                response = f'{location}에 있는 카페 정보\n{cafe_list}'
            else:
                response = '해당 지역의 카페 정보를 검색할 수 없습니다.'

        elif user_intent == '관광지':
            tour_list = get_tour_info(location)
            location = ' '.join(location)

            # tour_list가 빈 문자열이 아니거나 특정 조건을 만족하는 경우
            # if tour_list and not tour_list.startswith('None'):
            if tour_list:
                response = f'{location}에 있는 관광지 정보\n{tour_list}'
            else:
                response = '해당 지역의 관광지 정보를 검색할 수 없습니다.'

        else:
            response = '이해하지 못했어요.'
        return response

    def main(self):
        p = Preprocess(word2index_dic='E:/ai_chatbot/train_tools/dict/chatbot_dict.bin',
                       userdic='E:/ai_chatbot/util/user_dic.tsv')

        intent = IntentModel(model_name='E:/ai_chatbot/models/intent/intent_model.h5', preprocess=p)

        myquery = input()
        predict = intent.predict_class(myquery)
        myintent = intent.label[predict]

        result = Response.generate_response(self, myintent, myquery)

        return result

if __name__ == "__main__":
    Response.main(self)

# myquery_restaurant = '동성로에 갈만한 음식점 좀 알려줘'
# predict = intent.predict_class(myquery_restaurant)
# myintent_restaurant = intent.label[predict]
#
# result_restaurant = Response.generate_response(self, myintent_restaurant, myquery_restaurant)
# print(result_restaurant)
#
# print("="*30)
#
# myquery_cafe = '부천에 좋은 카페 알려줘'
# predict = intent.predict_class(myquery_cafe)
# myintent_cafe = intent.label[predict]
#
# result_cafe = Response.generate_response(self, myintent_cafe, myquery_cafe)
# print(result_cafe)
#
# print("="*30)
#
# myquery_tour = '부산에 갈만한 관광지 알려줘'
# predict = intent.predict_class(myquery_tour)
# myintent_tour = intent.label[predict]
#
# result_tour = Response.generate_response(self, myintent_tour, myquery_tour)
# print(result_tour)

