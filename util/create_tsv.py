import csv

with open('E:/ai_chatbot/util/user_dic.tsv', 'w', encoding='utf-8', newline='') as f:
    cw = csv.writer(f, delimiter='\t')
    # 지역
    # 수도권
    # cw.writerow(['keyword', 'label'])
    cw.writerow(['서울', 'NNP'])
    cw.writerow(['서울시', 'NNP'])
    cw.writerow(['서울특별시', 'NNP'])
    cw.writerow(['경기', 'NNP'])
    cw.writerow(['경기도', 'NNP'])
    cw.writerow(['인천', 'NNP'])
    cw.writerow(['인천시', 'NNP'])
    cw.writerow(['인천광역시', 'NNP'])

    # 강원도
    cw.writerow(['강원', 'NNP'])
    cw.writerow(['강원도', 'NNP'])

    # 충청도
    cw.writerow(['충청', 'NNP'])
    cw.writerow(['충청도', 'NNP'])
    cw.writerow(['충북', 'NNP'])
    cw.writerow(['충남', 'NNP'])
    cw.writerow(['충청북도', 'NNP'])
    cw.writerow(['충청남도', 'NNP'])
    cw.writerow(['대전', 'NNP'])
    cw.writerow(['대전시', 'NNP'])
    cw.writerow(['대전광역시', 'NNP'])

    # 전라도
    cw.writerow(['전라', 'NNP'])
    cw.writerow(['전라도', 'NNP'])
    cw.writerow(['전북', 'NNP'])
    cw.writerow(['전남', 'NNP'])
    cw.writerow(['전라북도', 'NNP'])
    cw.writerow(['전라남도', 'NNP'])
    cw.writerow(['광주', 'NNP'])
    cw.writerow(['광주시', 'NNP'])
    cw.writerow(['광주광역시', 'NNP'])

    # 경상도
    cw.writerow(['경상', 'NNP'])
    cw.writerow(['경상도', 'NNP'])
    cw.writerow(['경북', 'NNP'])
    cw.writerow(['경남', 'NNP'])
    cw.writerow(['경상북도', 'NNP'])
    cw.writerow(['경상남도', 'NNP'])
    cw.writerow(['울산', 'NNP'])
    cw.writerow(['울산시', 'NNP'])
    cw.writerow(['울산광역시', 'NNP'])
    cw.writerow(['대구', 'NNP'])
    cw.writerow(['대구시', 'NNP'])
    cw.writerow(['대구광역시', 'NNP'])
    cw.writerow(['부산', 'NNP'])
    cw.writerow(['부산시', 'NNP'])
    cw.writerow(['부산광역시', 'NNP'])

    # 제주
    cw.writerow(['제주', 'NNP'])
    cw.writerow(['제주도', 'NNP'])
    cw.writerow(['제주특별자치도', 'NNP'])

    # 식당
    cw.writerow(['음식점', 'NNG'])
    cw.writerow(['식당', 'NNG'])
    cw.writerow(['음식', 'NNG'])
    cw.writerow(['밥', 'NNG'])
    cw.writerow(['맛집', 'NNG'])
    cw.writerow(['음식', 'NNG'])
    cw.writerow(['식사', 'NNG'])
    cw.writerow(['아침', 'NNG'])
    cw.writerow(['저녁', 'NNG'])
    cw.writerow(['점심', 'NNG'])
    cw.writerow(['밥집', 'NNG'])
    cw.writerow(['식사', 'NNG'])

    # 카페
    cw.writerow(['빵', 'NNG'])
    cw.writerow(['카페', 'NNP'])
    cw.writerow(['커피', 'NNP'])
    cw.writerow(['디저트', 'NNP'])
    cw.writerow(['음료', 'NNG'])

    # 관광지
    cw.writerow(['관광', 'NNG'])
    cw.writerow(['장소', 'NNG'])
    cw.writerow(['명소', 'NNG'])
    
    # 키워드와 레이블 추가 예정
