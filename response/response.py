import self
from konlpy.tag import Okt
import pandas as pd

from util.Preprocess import Preprocess
from models.intent.IntentModel import IntentModel

class Response:
    def __init__(self):
        self.okt = Okt()
        self.p = Preprocess(word2index_dic='E:/ai_chatbot/train_tools/dict/chatbot_dict.bin',
                            userdic='E:/ai_chatbot/util/user_dic.tsv')
        self.intent = IntentModel(model_name='E:/ai_chatbot/models/intent/intent_model.h5', preprocess=self.p)

    def extracted_region(self, user_query):
        # 형태소 분석
        morphs = self.okt.pos(user_query)

        location_keywords = ['시', '군', '구', '동', '면', '읍', '리']

        # 제외할 관련 키워드 리스트
        keywords = ['밥', '음식점', '식당', '맛집', '음식', '식사', '아침', '점심', '저녁', '밥집', '식사', '좀', '근처', '주변', '관광지', '관광',
                    '명소', '카페', '커피', '케이크', '음료', '음료수', '추천', '계획', '박', '일','쇼핑', '백화점', '자연', '바다', '산', '산행',
                    '강', '계곡', '해수욕장', '호수', '박물관', '도서관', '서점', '유적지', '고궁', '궁', '전통', '레저', '스노쿨링', '스키', '래프팅',
                    '수영', '스카이다이빙', '낚시', '요트', '번지점프', '제트스키', '스노보드', '활동', '거', '것', '힐링', '타고', '해수욕', '정도',
                    '대략', '약', '약간', '등산', '탈', '해', '겨', '개', '곳']

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
                    extracted_regions[idx - 1] = (extracted_regions[idx - 1] + extracted_regions[idx])
                    del extracted_regions[idx]
            return extracted_regions
        else:
            return None

    def get_restaurant_info(self, location):
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
                           usecols=['사업장명', '소재지전체주소', '위도', '경도'])

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

        # 위도와 경도 정보를 추출하여 반환
        latitude = filtered_data['위도'].tolist()
        longitude = filtered_data['경도'].tolist()

        restaurant_info = filtered_data['사업장명'].tolist()
        return {'restaurant_info': restaurant_info[1:3], 'latitude': latitude[1:3], 'longitude': longitude[1:3]}

    def get_cafe_info(self, location):
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
                           usecols=['사업장명', '소재지전체주소', '위도', '경도'])

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

        # 위도와 경도 정보를 추출하여 반환
        latitude = filtered_data['위도'].tolist()
        longitude = filtered_data['경도'].tolist()

        cafe_info = filtered_data['사업장명'].tolist()
        return {'cafe_info': cafe_info[1:3], 'latitude': latitude[1:3], 'longitude': longitude[1:3]}

    def get_tour_info(self, location):
        if isinstance(location, list):
            if len(location) > 0:
                location = location[-1]
            else:
                return "지역 정보가 비어있습니다."
        elif location is None:
            return "지역을 찾을 수 없습니다."

        data = pd.read_csv('E:/ai_chatbot/변형데이터/전국관광지정보표준데이터.csv', encoding='cp949',
                           usecols=['소재지지번주소', '관광지명', '위도', '경도'])

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

        # 위도와 경도 정보를 추출하여 반환
        latitude = filtered_data['위도'].tolist()
        longitude = filtered_data['경도'].tolist()

        tour_info = filtered_data['관광지명'].tolist()
        # 결과 반환 시 위도와 경도 정보도 함께 반환
        return {'tour_info': tour_info[1:3], 'latitude': latitude[1:3], 'longitude': longitude[1:3]}
    def generate_response(self, user_intent, user_query):
        location = self.extracted_region(user_query)
        if location is None:
            return '지역명을 찾을 수 없습니다.'

        if user_intent == '식당':
            restaurant_list = self.get_restaurant_info(location)
            location = ' '.join(location)

            # restaurant_list가 빈 문자열이 아니거나 특정 조건을 만족하는 경우
            # if restaurant_list and not restaurant_list.startswith('None'):
            if restaurant_list:
                response = f'{location}에 있는 식당 정보\n{restaurant_list}'
            else:
                response = '해당 지역의 식당 정보를 검색할 수 없습니다.'

        elif user_intent == '카페':
            cafe_list = self.get_cafe_info(location)
            location = ' '.join(location)

            # cafe_list가 빈 문자열이 아니거나 특정 조건을 만족하는 경우
            # if cafe_list and not cafe_list.startswith('None'):
            if cafe_list:
                response = f'{location}에 있는 카페 정보\n{cafe_list}'
            else:
                response = '해당 지역의 카페 정보를 검색할 수 없습니다.'

        elif user_intent == '관광지':
            tour_list = self.get_tour_info(location)
            location = ' '.join(location)

            # tour_list가 빈 문자열이 아니거나 특정 조건을 만족하는 경우
            # if tour_list and not tour_list.startswith('None'):
            if tour_list:
                response = f'{location}에 있는 관광지 정보\n{tour_list}'
            else:
                response = '해당 지역의 관광지 정보를 검색할 수 없습니다.'

        # elif user_intent == '계획':

        else:
            response = '이해하지 못했어요.'
        return response

    def main(self, myquery):
        predict = self.intent.predict_class(myquery)
        myintent = self.intent.label[predict]

        result = self.generate_response(myintent, myquery)

        return result

# p = Preprocess(word2index_dic='E:/ai_chatbot/train_tools/dict/chatbot_dict.bin',
#                        userdic='E:/ai_chatbot/util/user_dic.tsv')
#
# intent = IntentModel(model_name='E:/ai_chatbot/models/intent/intent_model.h5', preprocess=p)
#
#
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

