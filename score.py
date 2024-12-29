import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
import pickle

# Load country flags data
sf = pd.read_csv('flags_iso.csv')

# Streamlit app header
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
    b = BeautifulSoup(r.text, 'html.parser')
    venue = b.find(class_='ds-flex ds-items-center').text.split(',')[1]
    
    # Initialize lists for data extraction
    list = []
    list1 = []
    list8 = []
    list9 = []
    list10 = []

    elements = b.find_all(class_='ds-cursor-pointer ds-pt-1')
    for i, element in enumerate(elements):
        if element.text.split('/'):
            if i % 2 != 0:
                list.append(element.text.split('/')[0])
                list1.append(element.text.split('/')[1].split('(')[0])
            else:
                list8.append(element.text.split('/')[0])
                list9.append(i / 2 + 1)
                list10.append(element.text.split('/')[1].split('(')[0])
                
    dict1 = {'inng1': list8, 'over': list9, 'wickets': list10}
    df1 = pd.DataFrame(dict1)

    dict = {'score': list, 'wickets': list1}
    df = pd.DataFrame(dict)

    # Convert and clean data
    df['score'] = pd.to_numeric(df['score'], errors='coerce').fillna(0)
    df['wickets'] = pd.to_numeric(df['wickets'], errors='coerce').fillna(0)
    df1['inng1'] = pd.to_numeric(df1['inng1'], errors='coerce').fillna(0)
    df1['wickets'] = pd.to_numeric(df1['wickets'], errors='coerce').fillna(0)

    # Initialize previous wickets column
    df1['previous_wickets'] = df1['wickets'].shift(1).fillna(0)

    # Perform subtraction safely
    df1['wic'] = df1['wickets'] - df1['previous_wickets']

    # Additional calculations
    df['target'] = df['score'].max() + 1  # Example target
    df['runs_left'] = df['target'] - df['score']
    df['crr'] = df['score'] / df['over']
    df['rrr'] = df['runs_left'] / (50 - df['over'])
    df['balls_left'] = 300 - (df['over'] * 6)
    df['runs'] = df['score'].diff().fillna(0)
    df['last_10'] = df['runs'].rolling(window=10).sum().fillna(0)
    df['wickets_in_over'] = df['wickets'].diff().fillna(0)
    df['last_10_wicket'] = df['wickets_in_over'].rolling(window=10).sum().fillna(0)

    # Display the scorecard
    o = int(o)
    if o != 50:
        col1, col2 = st.columns([1, 1])
        with col1:
            bowling_team = "Bowling Team"  # Replace with dynamic value
            batting_team = "Batting Team"  # Replace with dynamic value
            st.write(f"**{bowling_team}** vs **{batting_team}**")
        with col2:
            st.write(f"Target: {df['target'].max()}")
            st.write(f"Runs Left: {df['runs_left'].iloc[o]}")
    
    # Display plots
    fig = go.Figure(data=[
        go.Scatter(x=df1['over'], y=df1['inng1'], line_width=3, line_color='red', name="Innings 1"),
        go.Scatter(x=df['over'], y=df['score'], line_width=3, line_color='green', name="Innings 2")
    ])
    fig.update_layout(title='Score Comparison', xaxis_title='Over', yaxis_title='Score')
    st.plotly_chart(fig)

