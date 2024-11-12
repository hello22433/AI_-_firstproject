import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import time

def get_stock_buy_recommendation(stock_name):
    """
    Fetch news headlines and provide stock buy recommendations
    """
    # Add a spinner while processing
    with st.spinner(f'분석 중... {stock_name}에 대한 정보를 가져오는 중입니다.'):
        url = f'https://search.naver.com/search.naver?ie=utf8&sm=nws_hty&query={stock_name}'
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract headlines
        headlines = [headline.get_text() for headline in soup.find_all('a', class_='news_tit')[:10]]

        # Initialize sentiment analyzer
        # sentiment_analyzer = pipeline('sentiment-analysis')
        sentiment_analyzer = pipeline('sentiment-analysis', model="snunlp/KR-FinBert")

        # Analyze sentiments
        results = []
        positive_count = 0
        negative_count = 0

        for headline in headlines:
            sentiment = sentiment_analyzer(headline)[0]['label']

            if sentiment == 'POSITIVE':
                sentiment_emoji = '😊'
                positive_count += 1
            elif sentiment == 'NEGATIVE':
                sentiment_emoji = '😰'
                negative_count += 1
            else:
                sentiment_emoji = '😐'

            results.append((headline, sentiment_emoji))

        # Determine recommendation
        if positive_count > negative_count:
            buy_recommendation = f'{stock_name}을(를) 매수하세요 😊'
        elif positive_count < negative_count:
            buy_recommendation = f'{stock_name}을(를) 매수하지 마세요 😰'
        else:
            buy_recommendation = f'{stock_name}에 대해 중립적인 입장입니다 😐'

        return results, buy_recommendation
    
    
    

import numpy as np
import matplotlib.pyplot as plt
    
# LSTM 주가예측
def LSTM_pre(stock_name) :
    import yfinance as yf
    from sklearn.model_selection import train_test_split

    ticker = stock_name
    if int(ticker):
        ticker = f'{stock_name}.KS'
    else :
        ticker = stock_name
        
    data = yf.download(ticker, start="2020-01-01", end="2023-01-01")

    close_prices = data["Close"].values
    close_prices = close_prices.reshape(-1,1)

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))
    close_prices_scaled = scaler.fit_transform(close_prices)


    def create_windowed_data(data, window_size=60) :
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i+window_size])
            y.append(data[i+window_size])
        return np.array(X), np.array(y)

    window_size = 60
    X, y = create_windowed_data(close_prices_scaled, window_size)

    X = X.reshape((X.shape[0], X.shape[1], 1))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, shuffle=False)

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.optimizers import Adam


    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(window_size, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

    predicted_stock_price = model.predict(X_test)

    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
    y_test = scaler.inverse_transform(y_test.reshape(-1,1))

    # Streamlit으로 시각화
    fig, ax = plt.subplots()
    ax.plot(y_test, color="blue", label="Actual Stock Price")
    ax.plot(predicted_stock_price, color="red", label="Predicted Stock Price")
    ax.set_xlabel("Time")
    ax.set_ylabel("Stock Price")
    ax.legend()

    # Streamlit을 통해 그래프 출력
    st.pyplot(fig)

# # Set up the Streamlit page
# st.set_page_config(page_title="주식 투자 의견 분석기", page_icon="📈")

# # Add title and description
# st.title("📈 주식 투자 의견 분석기")
# st.markdown("""
# 이 앱은 네이버 뉴스 헤드라인을 분석하여 주식 투자 의견을 제공합니다.
# """)

# # Create input field
# stock_name = st.text_input("종목명을 입력하세요:", placeholder="예: 삼성전자")

# # Create analyze button
# if st.button("투자의견 분석"):
#     if stock_name:
#         try:
#             # Get recommendation
#             headline_results, buy_recommendation = get_stock_buy_recommendation(stock_name)

#             # Display results
#             st.subheader(f"{stock_name}에 대한 분석 결과")

#             # Display buy recommendation in a highlighted box
#             st.info(buy_recommendation)

#             # Display headlines in an expandable section
#             with st.expander("뉴스 헤드라인 상세 분석 보기"):
#                 for headline, emoji in headline_results:
#                     st.write(f"{emoji} {headline}")

#         except Exception as e:
#             st.error(f"오류가 발생했습니다: {str(e)}")
#     else:
#         st.warning("종목명을 입력해주세요.")

# # Add footer
# st.markdown("---")
# st.markdown("Made with ❤️ using Streamlit")