
#!/usr/bin/env python
# coding: utf-8

# In[14]:
from emotion_economics import get_stock_buy_recommendation, LSTM_pre
import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import pipeline

from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
import time
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support import expected_conditions as EC

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np


# In[31]:

# Set up the Streamlit page
st.set_page_config(page_title="주식 투자 의견 분석기", page_icon="📈")

# Add title and description
st.title("📈 주식 투자 의견 분석기")
st.markdown("""
이 앱은 네이버 뉴스 헤드라인을 분석하여 주식 투자 의견을 제공합니다.
""")

# Create input field
stock_name = st.text_input("티커를 입력해주세요:", placeholder="예: 005930")
stock_name = str(stock_name)

# Create analyze button
if st.button("투자의견 분석"):
    if stock_name:
        try:
            # Get recommendation
            headline_results, buy_recommendation = get_stock_buy_recommendation(stock_name)

            # Display results
            st.subheader(f"{stock_name}에 대한 분석 결과")

            # Display buy recommendation in a highlighted box
            st.info(buy_recommendation)

            # Display headlines in an expandable section
            with st.expander("뉴스 헤드라인 상세 분석 보기"):
                for headline, emoji in headline_results:
                    st.write(f"{emoji} {headline}")
            
            LSTM_pre(stock_name)

        except Exception as e:
            st.error(f"오류가 발생했습니다: {str(e)}")
    else:
        st.warning("종목명을 입력해주세요.")
        
        
        



#검색 아이콘 찾기. [아이콘이 없어 필요X]
# search_icon = driver.find_element(By.XPATH, '//*[@id="__next"]/header/div[1]/section/div[2]/div[1]/button/svg/path')
# print(search_icon.text)
options = Options()
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36")
options.add_argument("--headless")  # 브라우저 창을 띄우지 않음
options.add_argument("--no-sandbox")  # 리눅스 환경에서 필수인 경우가 많음
options.add_argument("--disable-dev-shm-usage")  # 리눅스 환경에서 필수인 경우가 많음
options.add_argument("--disable-gpu")  # GPU 가속 비활성화 (일부 환경에서 필요)
options.add_argument("--window-size=1920,1080")  # 브라우저 창 크기 설정

driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
url = "http://data.krx.co.kr/contents/MDC/MAIN/main/index.cmd"
driver.get(url)

# 페이지 로딩시간
WebDriverWait(driver, 10).until(
    lambda driver: driver.execute_script('return document.readyState') == 'complete'
)
time.sleep(2) # 페이지 로딩시간 안전장치


#검색 버튼 막는 창 끄기 
try : 
#     search_ad_button = WebDriverWait(driver, 20).until(
#     EC.visibility_of_element_located((By.XPATH, 'jsDetailLayer_MDCMAIN003"]/div[3]/div/button'))
# )
    search_ad_button = driver.find_element(By.XPATH, '//*[@id="jsDetailLayer_MDCMAIN003"]/div[3]/div/button')
    search_ad_button2 = driver.find_element(By.XPATH, '//*[@id="jsDetailLayer_MDCMAIN004"]/div[1]/button')
    search_ad_button2.click()
    search_ad_button.click()
except :
    pass
    
#검색 인풋 찾기
# search_input = WebDriverWait(driver, 20).until(
#     EC.visibility_of_element_located((By.XPATH, '//*[@id="jsTotSch"]'))
# )
search_input = driver.find_element(By.XPATH, '//*[@id="jsTotSch"]')
#검색 버튼 찾기
search_button = driver.find_element(By.XPATH, '//*[@id="jsTotSchBtn"]')
#검색 항목 변경을 위해서 여기서 종목이름 변경

#종목이름
find_name = stock_name

#검색 인풋 넣기
def func_search_input(x) :
    search_input.send_keys(x)
    search_button.click()
func_search_input(find_name)
#정보 가져오기

time.sleep(2)
#불러오기 정보 토큰
find_stock = '//*[@id="isuInfoTitle"]/span'
find_incdec = '//*[@id="isuInfoTitle"]/span/dfn'
find_start = '//*[@id="isuInfoBind"]/table/tbody/tr[1]/td[1]'
find_high = '//*[@id="isuInfoBind"]/table/tbody/tr[2]/td[1]'
find_low = '//*[@id="isuInfoBind"]/table/tbody/tr[3]/td[1]'

#토큰 불러오기
sear_name = find_name
sear_stock = driver.find_element(By.XPATH, find_stock)
sear_incdec = driver.find_element(By.XPATH, find_incdec)
sear_start = driver.find_element(By.XPATH, find_start)
sear_high = driver.find_element(By.XPATH, find_high)
sear_low = driver.find_element(By.XPATH, find_low)


#주가가 바로 추출이 안 되므로 전체str에서 추출.
sear_stock_str = ''
for i in sear_stock.text :
    if i=='▼' or i=='▲' or i=='(' or i=='-':
        break;
    sear_stock_str += i

#요구 정보 추출
print("종목이름 :", sear_name)
print("주가 :", sear_stock_str)
print("주가증감 :", sear_incdec.text)
print("시가 :", sear_start.text)
print("고가 :", sear_high.text)
print("저가 :", sear_low.text)

with st.expander("주가 정보 자세히 보기"): 
    st.write(f"종목이름 : {sear_name}")
    st.write(f"주가 : {sear_stock_str}")
    st.write(f"주가증감 : {sear_incdec.text}")
    st.write(f"시가 : {sear_start.text}")
    st.write(f"고가 : {sear_high.text}")
    st.write(f"저가 : {sear_low.text}")


#챗지피티
from openai import OpenAI

myapikey = "여기에 key를 입력하세요"

client = OpenAI(api_key = myapikey)

response = client.chat.completions.create(
    model = 'gpt-4o-mini',
    messages = [
        {"role":"system", "content":"너는 스마트한 금융 정보 제공 AI 챗봇이야."},
        {"role":"user", "content":f"실시간 검색을 이용해서 종목이름인 {sear_name}과, 주가인 {sear_stock_str}과, 주가증감인 {sear_incdec.text}과, 시가인 {sear_start.text}과, 고가인 {sear_high.text}와, 저가인 {sear_low.text}를 참고해서 {sear_name}의 정량적인 분석을 해줘"}
    ],
    max_tokens=500
)

## 챗지피티를 활용한 정량적인 분석
gpt_response = response.choices[0].message.content
with st.expander("주가 정보 정량적인 분석"): 
    st.write(gpt_response)


# 워드클라우드에 시각화할 정보(네이버 검색 정보)

import requests
from bs4 import BeautifulSoup
import urllib.parse

def get_naver_autocomplete(query, limit):
    # 네이버 자동완성 URL 생성
    base_url = "https://ac.search.naver.com/nx/ac"
    params = {
        'q': query,
        'con': '1',
        'frm': 'nv',
        'ans': '2',
        'r_format': 'json',
        'r_enc': 'UTF-8',
        'r_unicode': '0',
        'r_escape': '1',
        'st': '100'
    }
    
    response = requests.get(base_url, params=params)
    
    # 응답이 성공적일 경우 JSON 데이터 파싱
    if response.status_code == 200:
        data = response.json()
        
        # 자동완성 키워드 리스트 추출
        suggestions = [item[0] for item in data['items'][0]][:limit]
        return suggestions
    else:
        print("자동완성 키워드를 가져오는데 실패했습니다.")
        return []


# 챗 지피티 코드 재활용하여 네이버검색워딩과 연관된 워딩 찾아내기
response = client.chat.completions.create(
    model = 'gpt-4o-mini',
    messages = [
        {"role":"system", "content":"너는 스마트한 금융 정보 제공 AI 챗봇이야."},
        {"role":"user", "content":f"종목이름인 {sear_name}와 연관된 단어만을 20개만 띄어쓰기로 알려줘"}
    ],
    max_tokens=500
)

gpt_response = response.choices[0].message.content

# 챗 지피티로 찾아낸 연관워딩을 각각 네이버 연관 검색어 알아내기
gtext = ''
samsung_related_word = gpt_response.split(' ')
for i in samsung_related_word : 
    query = i
    keywords = get_naver_autocomplete(query, limit=5)
    print("자동완성 키워드:", " ".join(keywords))
    gtext += " "
    gtext += " ".join(keywords)


# In[517]:

import nltk
from nltk.tokenize import wordpunct_tokenize
from nltk.tag import pos_tag

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


# In[519]:

import os #운용체제 기능 도입
from collections import Counter #빈도수 위해 도입
import re #정규식 모듈
from nltk.corpus import stopwords #불용어 사전

# In[521]:

from konlpy.tag import Kkma
from konlpy.tag import Hannanum
from wordcloud import WordCloud, STOPWORDS # STOPWORDS에 영어 불용어만 들어 있으므로 사실 필요 없음
from PIL import Image

# In[523]:


# trump_mask = np.array(Image.open("real_black_trump.png"))
# trump_mask[0]


# In[524]:

import requests
from io import BytesIO

# 이미지를 네이버에서 불러오기
def get_image_from_naver(query, display=1):
    # 네이버 API 키 (발급받은 ID와 Secret을 여기에 입력하세요)
    client_id = "여기에 id key"
    client_secret = "여기에 pw key"
    url = "https://openapi.naver.com/v1/search/image"
    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret
    }
    params = {
        "query": query,  # 검색할 키워드
        "display": display,  # 가져올 이미지 수
        "start": 1,
        "sort": "sim"
    }

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        items = response.json().get('items')
        if items:
            # 첫 번째 이미지 URL 가져오기
            image_url = items[0].get("link")
            # 이미지 데이터를 가져와서 변수로 저장
            image_response = requests.get(image_url)
            if image_response.status_code == 200:
                image_data = Image.open(BytesIO(image_response.content))
                return image_data
            else:
                print("이미지 데이터 다운로드 실패")
        else:
            print("이미지를 찾을 수 없습니다.")
    else:
        print("API 요청 실패:", response.status_code)

# 이미지를 네이버에서 불러와 변수화
query = sear_name
image_data = get_image_from_naver(query)
st.write(image_data)

# 이미지 확인 테스트
# if image_data:
#     image_data.show()  # 이미지 확인


# In[525]:

# 이미지 추출 중간 확인
# plt.figure(figsize=(15,8))
# plt.imshow(image_data, cmap=plt.cm.gray, interpolation='bilinear')
# plt.axis('off')
# plt.show()


# In[526]:

import numpy as np
import re

# 이미지 워드클라우드를 위한 전처리 함수
def preprocess_image_for_wordcloud(image):
    # image가 numpy 배열 형식이라면 PIL 이미지로 변환
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # 이미지를 흑백으로 변환
    image = image.convert("L")  # 흑백 변환
    
    # 이진화 (픽셀 값이 128보다 작으면 0, 아니면 255로 변환)
    image = image.point(lambda x: 0 if x < 128 else 255, '1')

    # 이미지 크기 전환
    # image = image.resize((800, 800))
    
    # 결과 이미지 확인을 위한 출력 (선택적으로 사용)
    # image.show()  # 이 부분을 사용하여 전처리된 이미지를 시각적으로 확인할 수 있습니다.
    mask = np.array(image)
    return mask

# 이미지를 워클라우딩을 위해 전처리한 후 변수화
Mask = preprocess_image_for_wordcloud(image_data)


# 전처리된 마스크 확인 (이미지로 출력하거나 마스크 배열을 확인)
# print(Mask)



# In[527]:

def wordCloud_make(gtext) : 
    wc = WordCloud(
        background_color = 'white',
        max_words=2000,
        mask=Mask,
        contour_width=3,
        contour_color="steelblue",
        font_path = 'NanumGothicEco.ttf'
    )
    # text 표준화
    import re
    gtext = re.sub(r'[^\w\s]', '', gtext) 

    # 불용어 지정
    x_words = [
        '의', '에', '입니다', '이다', '있습니다', '수', '뜻', '영어로', '및', '400원', '400원은', '400원으로', '400원을'
    ]
    # text에서 불용어 걸러내기
    gtext = gtext.split(' ')
    bucket = ''
    for word in gtext :
        if not word in x_words:
            bucket += ' '
            bucket += word
            
    # 워드클라우드 생성
    wc.generate(bucket)
    
    # word_list = list(wc.words_.keys())
    # word_list[0:10]
    
    # 워드클라우드 출력
    fig = plt.figure(figsize=(15,8))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    st.write(fig)

wordCloud_make(gtext)


# In[529]:


