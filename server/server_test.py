import requests
import json

url = 'http://localhost:5000/generate_response'
headers = {'Content-Type': 'application/json'}
data = {'user_query': '서울에 있는 맛집 추천해줘'}

response = requests.post(url, headers=headers, data=json.dumps(data))
print(response.json())