from konlpy.tag import Okt
import pandas as pd
import requests

def extract_location(user_query):
    df = pd.read_csv('E:/ai_chatbot/변형데이터/RegionCode.csv',
                      usecols=['코드명', '소재지'])
    location_list = df['소재지'].unique().tolist()

    okt = Okt()
    nouns = okt.nouns(user_query)
    for noun in nouns:
        if noun in location_list:
            return noun
    return None

import pandas as pd

def extract_code(location):
    df = pd.read_csv('E:/ai_chatbot/변형데이터/RegionCode.csv',
                     usecols=['코드명', '소재지'])
    
    # location 변수가 문자열인지 확인하고, 문자열이 아닐 경우 문자열로 변환
    if not isinstance(location, str):
        location = str(location)
    
    # str.contains 메서드 사용 시 na=False 옵션 추가
    filtered_df = df[df['소재지'].str.contains(location, na=False)]

    if filtered_df.empty:
        return []
    return filtered_df['코드명'].tolist()

def get_restaurant_info(location):
    data = pd.read_csv('E:/ai_chatbot/변형데이터/RegionCode.csv',
                       usecols=['코드명', '소재지'])
    '''
    식당/카페 데이터에 지역코드 추가
    
    '''

class Response:
    def generate_response(intent, user_query):
        location = extract_location(user_query)
        # code = extract_code(location)
        if intent == "식당":
            restaurant_info = get_restaurant_info(location)
            response = f"근처 음식점 정보를 보여드리겠습니다.\n{restaurant_info}"
        # elif intent == "카페":
        #     cafe_info = get_cafe_info(location)
        #     response = f"근처 카페 정보를 보여드리겠습니다.\n{cafe_info}"
        # elif intent == "관광지":
        #     tour_info = get_tour_info(location)
        #     response = f"{location}의 관광지를 보여드리겠습니다.\n{tour_info}"
        # else:
        #     response = "죄송해요, 이해하지 못했어요."
        return response
    
print(Response.generate_response("식당", "목동에 맛있는 집 있어?"))