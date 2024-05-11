from konlpy.tag import Okt
import pandas as pd
import re

# 사용자 쿼리에서 지역 이름을 찾고 해당 정보를 반환하는 함수
# def extract_location(query):
#     data = pd.read_csv("E:/ai_chatbot/변형데이터/지역명데이터.csv")
#     # 데이터에서 지역명을 추출하여 검색 패턴을 만듦
#     pattern = '|'.join(
#         set(data['SIDO_NM'].unique().tolist() + data['SGG_NM'].unique().tolist() + data['DONG_NM'].unique().tolist()))
#
#     # 사용자의 질의에서 지역명을 찾음
#     found_regions = re.findall(pattern, query)
#
#     # 찾은 지역명이 없다면, None 반환
#     if not found_regions:
#         return None
#
#     # 찾은 지역명에 해당하는 데이터를 필터링
#     result = data[(data['SIDO_NM'].isin(found_regions)) | (data['SGG_NM'].isin(found_regions)) | (
#         data['DONG_NM'].isin(found_regions))]
#
#     # 결과 데이터가 비어 있다면, None 반환
#     if result.empty:
#         return None
#
#     # 결과값 중 하나를 문자열로 반환
#     return found_regions[0]

# def extract_location(query):
#     data = pd.read_csv("E:/ai_chatbot/변형데이터/지역명데이터.csv")
#     # 데이터에서 지역명을 추출하여 검색 패턴을 만듦
#     pattern = '|'.join(
#         set(data['DONG_NM'].unique().tolist() + data['SGG_NM'].unique().tolist() + data['SIDO_NM'].unique().tolist()))
#
#     # 사용자의 질의에서 지역명을 찾음
#     found_regions = re.findall(pattern, query)
#
#     # 찾은 지역명이 없다면, None 반환
#     if not found_regions:
#         return None
#
#     # 찾은 지역명 중에서 가장 구체적인(길이가 가장 긴) 지역명을 반환
#     found_regions.sort(key=len, reverse=True)
#     return found_regions[0]

''' 얘가 그나마 나은 함수 근데 좀 문제가 있긴 함 '''
# def extract_location(query):
#     data = pd.read_csv("E:/ai_chatbot/변형데이터/지역명데이터.csv")
#     # 데이터에서 지역명을 추출하여 검색 패턴을 생성
#     all_locations = set(
#         data['DONG_NM'].unique().tolist() + data['SGG_NM'].unique().tolist() + data['SIDO_NM'].unique().tolist())
#
#     # 지역명을 정렬하여 가장 긴 이름이 먼저 오도록 함
#     all_locations = sorted(all_locations, key=len, reverse=True)
#
#     # 패턴 매칭을 위해 지역명 중 공백을 제거하고 정규식을 생성
#     pattern = '|'.join(re.escape(location.replace(" ", "")) for location in all_locations)
#
#     # 사용자의 질의에서 공백을 제거
#     query_without_spaces = query.replace(" ", "")
#
#     # 사용자의 질의에서 지역명을 찾음
#     found_regions = re.findall(pattern, query_without_spaces, flags=re.IGNORECASE)
#
#     # 찾은 지역명이 없다면, None 반환
#     if not found_regions:
#         return None
#
#     # 찾은 지역명이 여러 개인 경우, 원본 데이터에서 해당되는 지역명을 찾아서 반환
#     if len(found_regions) > 1:
#         # 찾은 지역명 리스트를 초기화
#         refined_locations = []
#         for found in found_regions:
#             # 원본 지역명 목록에서 정규식으로 찾은 지역명과 일치하거나 포함하는 지역명을 찾아 추가
#             for location in all_locations:
#                 if found.lower() in location.replace(" ", "").lower():
#                     refined_locations.append(location)
#                     break
#         found_regions = ''.join(refined_locations)
#     else:
#         # 단일 지역명인 경우, 원본 데이터에서 해당되는 지역명을 찾아서 반환
#         for location in all_locations:
#             if found_regions[0].lower() in location.replace(" ", "").lower():
#                 found_regions = location
#                 break
#
#     return found_regions

# 사용자의 질문에서 지역명을 추출하는 함수
def extract_location(query):
    data = pd.read_csv("E:/ai_chatbot/변형데이터/지역명데이터.csv")
    all_locations = set(
        data['DONG_NM'].unique().tolist() + data['SGG_NM'].unique().tolist() + data['SIDO_NM'].unique().tolist())

    # Iterate over substrings of decreasing length
    for length in range(len(query), 0, -1):
        for start in range(0, len(query) - length + 1):
            substring = query[start:start + length]

            # Check for location match
            if substring in all_locations:
                # Additional filtering criteria
                if substring == "대구" and not substring.startswith("해운대"):
                    return substring

    return None

# 식당 정보 불러오기
def get_restaurant_info(location):
    # 리스트가 전달되었다면 첫 번째 항목을 사용
    if isinstance(location, list):
        if len(location) > 0:
            location = location[0]
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

    restaurant_info = '\n'.join(filtered_data['사업장명'].tolist())
    return restaurant_info

class Response:
    @staticmethod
    def generate_response(intent, user_query):
        if intent == "식당":
            location = extract_location(user_query)
            print(location)
            if location is None:  # location이 None인 경우 처리
                return "지역명을 찾을 수 없습니다."

            restaurant_info = get_restaurant_info(location)

            # restaurant_info가 빈 문자열이 아니거나 특정 조건을 만족하는 경우
            if restaurant_info and not restaurant_info.startswith("Cannot find"):
                response = f"{location}에 있는 음식점 정보입니다\n{restaurant_info}"
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

result = Response.generate_response("식당", "대구에 맛있는 집 있어?")
print(result)