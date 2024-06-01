import re
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
        self.keyword_to_sub_category = {
            '자연': ['국립공원', '도립공원', '군립공원', '산', '자연생태관광지', '자연휴양림', '수목원', '폭포', '계곡', '약수터', '해안절경', '해수욕장', '섬',
                   '항구', '포구', '등대', '호수', '강', '동굴', '바다', '해수욕', '해변'],
            '레저': ['수상레포츠', '항공레포츠', '수련시설', '경기장', '인라인', '자전거하이킹', '카트', '골프', '경마', '경륜', '카지노', '승마',
                   '스키', '스노보드', '스케이트', '썰매장', '수렵장', '사격장', '야영장', '오토캠핑장', '암벽등반', '서바이벌게임', 'ATV', 'MTB', '오프로드',
                   '번지점프', '트래킹', '윈드서핑', '제트스키', '카약', '카누', '요트', '스노쿨링', '스킨스쿠버다이빙', '민물낚시', '바다낚시', '수영', '래프팅',
                   '스카이다이빙', '초경량비행', '헹글라이딩', '패러글라이딩', '열기구', '복합 레포츠'],
            '식당': ['밥', '음식점', '식당', '맛집', '음식', '식사', '아침', '점심', '저녁', '밥집',],
            '카페': ['빵', '카페', '커피', '디저트', '음료', '케이크']
        }

    def extracted_region(self, user_query):
        morphs = self.okt.pos(user_query)
        location_keywords = ['시', '군', '구', '동', '면', '읍', '리']
        # 제외할 단어
        keywords = ['밥', '음식점', '식당', '맛집', '음식', '식사', '아침', '점심', '저녁', '밥집', '식사', '좀', '근처', '주변', '관광지', '관광',
                    '명소', '카페', '커피', '케이크', '음료', '음료수', '추천', '계획', '박', '일', '쇼핑', '백화점', '자연', '바다', '산', '산행',
                    '강', '계곡', '해수욕장', '호수', '박물관', '도서관', '서점', '유적지', '고궁', '궁', '전통', '레저', '스노쿨링', '스키', '래프팅',
                    '수영', '스카이다이빙', '낚시', '요트', '번지점프', '제트스키', '스노보드', '활동', '거', '것', '힐링', '타고', '해수욕', '정도',
                    '대략', '약', '약간', '등산', '탈', '해', '겨', '개', '곳', '갈래', '노래', '타래']

        # sub_category 키워드를 제외할 단어에 추가
        for sublist in self.keyword_to_sub_category.values():
            keywords.extend(sublist)

        extracted_regions = []

        for word, tag in morphs:
            if tag in ['Noun', 'ProperNoun'] and word not in keywords:
                extracted_regions.append(word)

        if extracted_regions:
            for key in location_keywords:
                if key in extracted_regions:
                    idx = extracted_regions.index(key)
                    extracted_regions[idx - 1] = (extracted_regions[idx - 1] + extracted_regions[idx])
                    del extracted_regions[idx]
            return extracted_regions[0]
        else:
            return None

    def find_sub_category(self, user_query, extracted_regions):
        user_query_no_location = user_query
        for region in extracted_regions:
            user_query_no_location = user_query_no_location.replace(region, "")
        for main_category, sub_categories in self.keyword_to_sub_category.items():
            for sub_category in sub_categories:
                if sub_category in user_query_no_location:
                    return sub_category
        return None

    def extract_duration(self, user_query):
        duration_pattern = re.compile(r'(\d박\s*\d일|당일치기)')
        match = duration_pattern.search(user_query)
        if match:
            return match.group()
        else:
            return None

    def generate_response(self, user_intent, user_query):
        location = self.extracted_region(user_query)
        if location is None:
            return '지역명을 찾을 수 없습니다.'

        sub_category = self.find_sub_category(user_query, location)
        duration = self.extract_duration(user_query)

        # response = f'사용자 의도: {user_intent}, 지역명: {" ".join(location)}'

        if not sub_category:
            sub_category = None
        if not duration:
            duration = None
        # if sub_category:
        #     response += f', 세부 카테고리: {sub_category}'
        # else:
        #     sub_category = None
        #     response += f', 세부 카테고리: {sub_category}'
        #
        # if duration:
        #     response += f', 여행 기간: {duration}'
        # else:
        #     duration = None
        #     response += f', 여행 기간: {duration}'

        return user_intent, location, sub_category, duration

    # def response_to_json(self, user_intent, location, sub_category, duration):
    #     response_dict = {
    #         "user_intent": user_intent,
    #         "location": location,
    #         "sub_category": sub_category,
    #         "duration": duration
    #     }
    #     return json.dumps(response_dict, ensure_ascii=False)

    def main(self, myquery):
        predict = self.intent.predict_class(myquery)
        myintent = self.intent.label[predict]
        # location = self.extracted_region(myquery)
        # sub_category = self.find_sub_category(myquery, location)
        # duration = self.extract_duration(myquery)

        return self.generate_response(myintent, myquery)