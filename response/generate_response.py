import self
from konlpy.tag import Okt
import pandas as pd

from models.intent.IntentModel import IntentModel
from util.Preprocess import Preprocess

data = pd.read_excel('E:/ai_chatbot/원본데이터/한국관광공사_국문_서비스분류코드_v4.2.xlsx', header=4)

# 열 이름 확인
print(data.columns)


class Response:
    def __init__(self):
        self.okt = Okt()
        self.p = Preprocess(word2index_dic='E:/ai_chatbot/train_tools/dict/chatbot_dict.bin',
                            userdic='E:/ai_chatbot/util/user_dic.tsv')
        self.intent = IntentModel(model_name='E:/ai_chatbot/models/intent/intent_model.h5', preprocess=self.p)


        # 소분류와 대분류 사전 생성
        self.sub_category_dict = {row['소분류']: row['소분류 (cat3)'] for _, row in data.iterrows()}
        self.main_category_dict = {row['소분류']: row['대분류'] for _, row in data.iterrows()}

        # 키워드와 코드 매핑 사전 (추가)
        self.keyword_to_sub_category = {
            '자연': ['국립공원', '도립공원', '군립공원', '산', '자연생태관광지', '자연휴양림', '수목원', '폭포', '계곡', '약수터', '해안절경', '해수욕장', '섬', '항구/포구', '등대', '호수', '강', '동굴'],
            '역사': ['고궁', '성', '문', '고택', '생가', '민속마을', '유적지/사적지', '사찰', '종교성지', '안보관광'],
            '레저': ['수상레포츠', '항공레포츠', '수련시설', '경기장', '인라인(실내 인라인 포함)', '자전거하이킹', '카트', '골프', '경마', '경륜', '카지노', '승마', '스키/스노보드', '스케이트', '썰매장', '수렵장', '사격장', '야영장,오토캠핑장', '암벽등반', '서바이벌게임', 'ATV', 'MTB', '오프로드', '번지점프', '스키(보드) 렌탈샵', '트래킹', '윈드서핑/제트스키', '카약/카누', '요트', '스노쿨링/스킨스쿠버다이빙', '민물낚시', '바다낚시', '수영', '래프팅', '스카이다이빙', '초경량비행', '헹글라이딩/패러글라이딩', '열기구', '복합 레포츠'],
            '쇼핑': ['5일장', '상설시장', '백화점', '면세점', '대형마트', '전문매장/상가', '공예/공방', '특산물판매점', '사후면세점']
            # 필요에 따라 추가 키워드와 소분류를 매핑
        }

        print("Sub-category dict:", self.sub_category_dict)
        print("Main-category dict:", self.main_category_dict)

        # # 키워드와 코드 매핑 사전
        # self.keyword_to_code = {
        #     '자연': ['자연', '바다', '산', '산행', '강', '계곡', '해수욕장', '호수'],
        #     '교육': ['박물관', '도서관', '서점', '유적지', '고귱', '궁', '전통'],
        #     '스포츠': ['레저', '스노쿨링', '스키', '래프팅', '수영', '스카이다이빙', '낚시', '요트', '번지점프', '제트스키', '스노보드'],
        #     '쇼핑': ['쇼핑', '백화점']
        # }
        #
        # # 코드별로 분류된 키워드
        # self.code_mapping = {
        #     'A01': '자연관광지',
        #     'A02': '역사관광지',
        #     'A03': '레저스포츠',
        #     'A04': '쇼핑'
        # }

    def extract_info(self, user_query):
        morphs = self.okt.pos(user_query)

        location_keywords = ['시', '군', '구', '동', '면', '읍', '리']
        period_keywords = ['1박 2일', '2박 3일', '3박 4일', '4박 5일', '당일치기']

        # 제외할 관련 키워드 리스트
        keywords = ['밥', '음식점', '식당', '맛집', '음식', '식사', '아침', '점심', '저녁', '밥집', '식사', '좀', '근처', '주변', '관광지', '관광',
                    '명소', '카페', '커피', '케이크', '음료', '음료수', '추천', '계획', '박', '일','쇼핑', '백화점', '자연', '바다', '산', '산행',
                    '강', '계곡', '해수욕장', '호수', '박물관', '도서관', '서점', '유적지', '고귱', '궁', '전통', '레저', '스노쿨링', '스키', '래프팅',
                    '수영', '스카이다이빙', '낚시', '요트', '번지점프', '제트스키', '스노보드', '활동', '거', '것', '힐링', '타고', '해수욕', '정도',
                    '대략', '약', '약간', '등산',]

        period = None
        location = []

        for word in period_keywords:
            if word in user_query:
                period = word

        for word, tag in morphs:
            if tag in ['Noun', 'ProperNoun'] and word not in keywords:
                location.append(word)

        # 후처리
        valid_locations = []
        if location:
            combined_location = []
            for word in location:
                combined_location.append(word)
                if any(keyword in word for keyword in location_keywords):
                    valid_locations.append(' '.join(combined_location))
                    combined_location = []

            if combined_location:
                valid_locations.append(' '.join(combined_location))

            if not valid_locations:
                valid_locations = location[:1]  # 추출된 명사들 중 첫 번째 명사를 사용

        return valid_locations, period

    def generate_response(self, user_query, user_intent):
        locations, period = self.extract_info(user_query)
        if not locations:
            return {'error': '지역명을 찾을 수 없습니다.'}
        if not period:
            return {'error': '여행 기간을 찾을 수 없습니다.'}

        location = ' '.join(locations)

        activity = user_intent

        # 사용자 의도에 따라 활동 코드를 결정
        if user_intent == '자연':
            activity_code = 'A0101'
        elif user_intent == '교육':
            activity_code = 'A0201'
        elif user_intent == '스포츠':
            activity_code = 'A03'
        elif user_intent == '쇼핑':
            activity_code = 'A0401'
        else:
            return {'error': '해당하는 활동을 찾을 수 없습니다.'}

        return {
            'location': location,
            'period': period,
            'activity': activity,
            'activity_code': activity_code
        }

    def main(self, user_query):
        predict = self.intent.predict_class(user_query)
        intent = self.intent.label[predict]

        result = self.generate_response(user_query, intent)

        return result


# Example usage:
response = Response()
print(response.main("1박 2일 정도 강원도 가서 스키 타고 싶어"))