import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import pandas as pd
from selenium.webdriver.chrome.options import Options
#%%
# 검색어
# info = input("검색할 내용을 말해주세요 : ")
info = input("검색할 내용을 말해주세요 : ")
translation = "한글로 번역"
# 디버그 구글로그인된 크롬을 준비
chrome_options = Options() ## 옵션 추가를 위한 준비
chrome_options.add_experimental_option("debuggerAddress", "127.0.0.1:9222") ## 디버깅 옵션 추가
driver = webdriver.Chrome(options=chrome_options)
driver.implicitly_wait(3)
#%%
# 검색한 여러 영상에 urls,titles 정보값을 얻음
# 주소
driver.get("https://www.youtube.com/")
youtube = driver.find_element(By.NAME,'search_query')
youtube.send_keys(info)
time.sleep(2)
youtube.send_keys(Keys.ENTER)
time.sleep(2)

if(True):
    y = 0
    y_step = 3000
    for k in range(1,3):
        y = y + y_step
        script = "window.scrollTo({0},{1})".format(0,y)
        driver.execute_script(script)
        time.sleep(3)
html = driver.page_source
soup = BeautifulSoup(html,'lxml') # 가져온 html을 lxml로 html문서로 변환
title = soup.select('a#video-title') # a태그에서 select 해서 
titles = []
urls = []
# 내가 필요한건 shorts가 아님으로 url에서 shorts란 단어를 제외하도록 설정
for tag in title:
    try:        
        name = tag['title']
        url = tag['href']
        if 'shorts' in url :
            pass
        else:
            titles.append(name)
            urls.append("https://www.youtube.com"+ url)
    except:
        pass

#%% 
# 반복문으로 url에 들어가서 summary요약을 저장
new_summary = []

for i in urls:
    driver.get(i)
    time.sleep(4)
    driver.find_element(By.ID, "yt_ai_summary_header_toggle").click()    
    driver.find_element(By.ID, "yt_ai_summary_header_summary").click()
    
    # summary 버튼 클릭
    time.sleep(4)
    driver.switch_to.window(driver.window_handles[-1])
    time.sleep(5)
    # driver.send_keys(Keys.ENTER)
    # 진행하다 보면 내용이 긴 작업은 아래 내려가는 버튼으로 스크롤을 내릴 필요가 있다.
    try:# 내용이 길 경우 버튼 클릭이 필요
        driver.find_element(By.XPATH, '//*[@id="__next"]/div[1]/div[2]/main/div[2]/div[1]/div/div/div/div/button').click()
        
    except:
        pass
    # 검색해서 나온 내용을 한글화
    # 기본
    GPT = driver.find_element(By.ID,'prompt-textarea')
    GPT.send_keys(translation)
    time.sleep(1)
    GPT.send_keys(Keys.ENTER)
    time.sleep(10)
    # 변수(문장이 긴 경우)
    try:
       driver.find_element(By.CSS_SELECTOR, "button:nth-child(3) > svg").click()
    except:
        pass
    time.sleep(5)
    # 요약한 내용을 리스트로 만들고 다시 리스트화 시켜서 개별화 시킴
    html_A = driver.page_source
    chat = BeautifulSoup(html_A,'lxml')
    list_A = chat.select('div:nth-child(5) > div > div > div.relative.flex.w-full.flex-col.agent-turn > div.flex-col.gap-1.md\:gap-3 > div.flex.flex-grow.flex-col.max-w-full > div > div > ul > li')
    list_A
    x = []
    for a in list_A:
        x.append(a.text)
    new_summary.append(x)
    driver.close() # 현재 창 종료
    driver.switch_to.window(driver.window_handles[-1])  #다시 이전 창(탭)이동 그래야 설정이 계속 적용됌
    time.sleep(1)
#%%
# 새탭으로 이동
organized_list = ['\n'.join(a) for a in new_summary] # 리스트들을 하나에 value로 묶음
search = []
for title,url in zip(titles,urls):
    searchs = [title,url]
    search.append(searchs)
info_name = pd.DataFrame(search,columns=["제목","링크"])
summary_count = pd.DataFrame(organized_list ,columns=["요약"]) 

other_df = info_name.iloc[:len(summary_count)]
concatenated_df = pd.concat([other_df,summary_count], axis=1)
concatenated_df.set_index('제목', inplace=True)
#%%

concatenated_df.to_csv(info + ".csv",  encoding = 'utf8')
concatenated_df.to_excel(info + ".xlsx")
#%%
'''
공부하면서 새롭게 알게 된 정보
- 새로 열은 탭으로 이동
- 웹 안에서 텍스트를 찾을때  ul>li 안에 있다는 점
- 실행했을때 여러가지 변수를 찾아서 고쳐보는 점
- 생각한거 처럼 크롬에 확장 프로그램을 이용해보기

전에 작업했던 내용중에 겹치는 건 복습에 도움이 되었고
새로 알게된 사실은 좋은 정보가 되었습니다.
이 자동화를 만든 이유는 나중에 빅데이터를 수집할 때 유튜브라는 사이트에서
데이터를 조금이라도 더 모을 수 있게 만들어 보고자 만들게 되었습니다.


추후에 작업한다면 영상들에 최신화를 넣는것도 괜찮을꺼같다.
'''
# 긴 내용들은 번역이안되서 나오기때문에 문제점을 찾아봐야될꺼같다 시간을 조정하던지


