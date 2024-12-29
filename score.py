import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from bs4 import BeautifulSoup
import requests

# Load country flags data
sf = pd.read_csv('flags_iso.csv')

# Streamlit app header
st.header('ODI MATCH ANALYSIS')
st.sidebar.header('Analysis')
selected_section = st.sidebar.radio('Select a Section:', 
                                     ('Score Comparison', 'Innings Progression', 'Win Probability', 'Current Predictor'))

# Define the function for Score Comparison
o = st.number_input('Over No. (Not Greater Than Overs Played in 2nd Innings)', value=50)
h = st.text_input('URL (ESPN CRICINFO >Select Match > Click On Overs)', 
                  placeholder='Enter match URL')

if not h or h == 'https://www.espncricinfo.com/series/...':
    st.write('Please enter a valid URL.')
else:
    r = requests.get(h)
    b = BeautifulSoup(r.text, 'html.parser')
    venue = b.find(class_='ds-flex ds-items-center').text.split(',')[1]

    # Initialize lists for data extraction
    list, list1, list8, list9, list10 = [], [], [], [], []

    elements = b.find_all(class_='ds-cursor-pointer ds-pt-1')
    for i, element in enumerate(elements):
        if i % 2 != 0:
            list.append(element.text.split('/')[0])
            list1.append(element.text.split('/')[1].split('(')[0])
        else:
            list8.append(element.text.split('/')[0])
            list9.append(i / 2 + 1)
            list10.append(element.text.split('/')[1].split('(')[0])

    # Create DataFrames
    dict1 = {'inng1': list8, 'over': list9, 'wickets': list10}
    df1 = pd.DataFrame(dict1)

    dict = {'score': list, 'wickets': list1, 'over': list9, 'venue': venue}
    df = pd.DataFrame(dict)
    df['score'] = df['score'].astype(int)

    # Data processing
    df['previous_wickets'] = df['wickets'].shift(1).fillna(0).astype(int)
    df['wic'] = df['wickets'] - df['previous_wickets']
    df['target'] = 300  # Example target
    df['runs_left'] = df['target'] - df['score']
    df['crr'] = df['score'] / df['over']
    df['rrr'] = (df['target'] - df['score']) / (50 - df['over'])

    # Truncate DataFrame to overs played
    df = df[df['over'] <= o]

    # Display the scorecard
    col1, col2 = st.columns([1, 1])
    with col1:
        st.write(f"Target: {df['target'].unique()[0]}")
    with col2:
        st.write(f"Runs left: {df['runs_left'].iloc[-1]}")

    # Plotly Graphs for Score Comparison
    fig = go.Figure(data=[
        go.Scatter(x=df1['over'], y=df1['inng1'], line=dict(width=3, color='red'), name='1st Innings'),
        go.Scatter(x=df['over'], y=df['score'], line=dict(width=3, color='green'), name='2nd Innings')
    ])

    # Adding wickets to the plot
    wicket_text = df['wic'].astype(str)
    wicket_y = df['score'] + df['wic']
    wicket_y[wicket_y == df['score']] = None  # Hide points for 0 wickets
    fig.add_trace(go.Scatter(
        x=df['over'], y=wicket_y, mode='markers', name='Wickets',
        marker=dict(color='green', size=8), text=wicket_text, textposition='top center'
    ))

    fig.update_layout(title='Score Comparison', xaxis_title='Over', yaxis_title='Score')

    st.write(fig)

    # Win Probability Section
    if selected_section == 'Win Probability':
        # Ensure temp_df and required calculations are defined
        if 'temp_df' in locals():
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=temp_df.iloc[10:, :]['end_of_over'], y=temp_df.iloc[10:, :]['win'], mode='lines',
                name=temp_df['batting_team'].unique()[0], line=dict(color='green', width=4)
            ))
            fig3.add_trace(go.Scatter(
                x=temp_df.iloc[10:, :]['end_of_over'], y=temp_df.iloc[10:, :]['lose'], mode='lines',
                name=temp_df['bowling_team'].unique()[0], line=dict(color='red', width=4)
            ))
            fig3.update_layout(title=f"Win Probability (Target: {df['target'].unique()[0]})", height=700)
            st.write(fig3)
        else:
            st.write("Win Probability calculations not available.")
