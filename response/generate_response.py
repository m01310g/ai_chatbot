from konlpy.tag import Okt
import pandas as pd
import requests

# 사용자 질문에서 지역 추출
def extract_location(user_query):
    df = pd.read_csv('E:/ai_chatbot/변형데이터/RegionCode.csv',
                      usecols=['코드명', '소재지'])
    location_list = df['소재지'].unique().tolist()

    okt = Okt()
    nouns = okt.nouns(user_query)
    extracted_locations = [noun for noun in nouns if noun in location_list]

    if extracted_locations:
        # 가장 긴 위치명을 반환
        return max(extracted_locations, key=len)
    return None


# 식당 정보 불러오기
def get_restaurant_info(location):
    if location is None:
        return "지역을 찾을 수 없습니다."
    
    data = pd.read_csv('E:/ai_chatbot/변형데이터/RestaurantData.csv',
                       usecols=['사업장명', '소재지전체주소'])
    # case=False를 사용하여 대소문자 구분 없이 검색합니다.
    filtered_data = data[data['소재지전체주소'].str.contains(location, case=False, na=False)]

    if filtered_data.empty:
        return "해당 지역의 식당 정보를 찾을 수 없습니다."
    
    # 데이터를 문자열로 변환하여 반환합니다.
    # 예시로, 사업장명만 반환하는 간단한 형태로 구성했습니다.
    restaurant_info = '\n'.join(filtered_data['사업장명'].tolist())
    return restaurant_info

class Response:
    def generate_response(intent, user_query):
        if intent == "식당":
            location = extract_location(user_query)
            restaurant_info = get_restaurant_info(location)
            
            # restaurant_info가 빈 문자열이 아니거나 특정 조건을 만족하는 경우
            if restaurant_info and not restaurant_info.startswith("Cannot find"):
                response = f"{location}에 있는 맛있는 집 정보입니다: {restaurant_info}"
            else:
                # restaurant_info가 빈 문자열이거나 오류 메시지를 포함하는 경우
                response = "죄송해요, 해당 지역의 식당 정보를 찾을 수 없어요."
        # 기타 의도에 대한 처리...
        else:
            response = "죄송해요, 이해하지 못했어요."
        # elif intent == "카페":
        #     cafe_info = get_cafe_info(location)
        #     response = f"근처 카페 정보를 보여드리겠습니다.\n{cafe_info}"
        # elif intent == "관광지":
        #     tour_info = get_tour_info(location)
        #     response = f"{location}의 관광지를 보여드리겠습니다.\n{tour_info}"
        # else:
        #     response = "죄송해요, 이해하지 못했어요."
        return response
    
print(Response.generate_response("식당", "서울특별시 동작구 사당3동에 맛있는 집 있어?"))