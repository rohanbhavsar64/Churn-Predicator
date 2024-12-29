import pandas as pd 
from bs4 import BeautifulSoup
import requests 
import streamlit as st 
import numpy as np

sf = pd.read_csv('flags_iso.csv')
st.header('ODI MATCH ANALYSIS')
st.sidebar.header('Analysis')
selected_section = st.sidebar.radio('Select a Section:', 
                                     ('Score Comparison', 'Innings Progression', 'Win Probability', 'Current Predictor'))

# Define the function for Score Comparison
o = st.number_input('Over No.(Not Greater Than Overs Played in 2nd Innings)') or 50
h = st.text_input('URL(ESPN CRICINFO >Select Match > Click On Overs)') or 'https://www.espncricinfo.com/series/icc-cricket-world-cup-2023-24-1367856/australia-vs-south-africa-2nd-semi-final-1384438/match-overs-comparison'

if h == 'https://www.espncricinfo.com/series/icc-cricket-world-cup-2023-24-1367856/australia-vs-south-africa-2nd-semi-final-1384438/match-overs-comparison':
    st.write('Enter Your URL')
else:
    r = requests.get(h)
    b = BeautifulSoup(r.text, 'html')
    venue = b.find(class_='ds-flex ds-items-center').text.split(',')[1]
    
    list = []
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    list5 = []
    list6 = []
    list7 = []
    list8 = []
    list9 = []
    list10 = []

    elements = b.find_all(class_='ds-cursor-pointer ds-pt-1')
    for i, element in enumerate(elements):
        if not element.text.split('/'):
            print(' ')
        else:
            if i % 2 != 0:
                list.append(element.text.split('/')[0])
                list1.append(element.text.split('/')[1].split('(')[0])

    for i, element in enumerate(elements):
        if element.text.split('/') is None:
            print(' ')
        else:
            if i % 2 == 0:
                list8.append(element.text.split('/')[0])
                list9.append(i / 2 + 1)
                list10.append(element.text.split('/')[1].split('(')[0])
                
    dict1 = {'inng1': list8, 'over': list9, 'wickets': list10}
    df1 = pd.DataFrame(dict1)

    for i in range(len(list)):
        list2.append(b.find_all(class_='ds-text-tight-s ds-font-regular ds-flex ds-justify-center ds-items-center ds-w-7 ds-h-7 ds-rounded-full ds-border ds-border-ui-stroke ds-bg-fill-content-prime')[i].text)
        list3.append(b.find(class_='ds-text-compact-m ds-text-typo ds-text-right ds-whitespace-nowrap').text.split('/')[0])
        list4.append(b.find_all('th', class_='ds-min-w-max')[1].text)
        list5.append(b.find_all('th', class_='ds-min-w-max')[2].text)
        list6.append(b.find(class_='ds-flex ds-items-center').text.split(',')[1])
        if o == 50:
            list7.append(b.find(class_='ds-text-tight-s ds-font-medium ds-truncate ds-text-typo').text.split(' ')[0])

    if o == 50:
        dict = {'batting_team': list5, 'bowling_team': list4, 'venue': list6, 'score': list, 'wickets': list1, 'over': list2, 'target': list3, 'winner': list7} 
    else:
        dict = {'batting_team': list5, 'bowling_team': list4, 'venue': list6, 'score': list, 'wickets': list1, 'over': list2, 'target': list3} 

    df = pd.DataFrame(dict)

    df['score'] = df['score'].astype('int')
    df1['inng1'] = df1['inng1'].astype('int')
    df1['wickets'] = df1['wickets'].astype('int')
    df1['previous_wickets'] = df1['wickets'].shift(1)
    df1['previous_wickets'].loc[0] = 0
    df1['wic'] = df1['wickets'] - df1['previous_wickets
