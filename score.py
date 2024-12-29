import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup

# Load country flags data
sf = pd.read_csv('flags_iso.csv')

# Streamlit app header
st.header('ODI MATCH ANALYSIS')
st.sidebar.header('Analysis')
selected_section = st.sidebar.radio('Select a Section:', 
                                     ('Score Comparison', 'Innings Progression', 'Win Probability', 'Current Predictor'))

# Define the function for Score Comparison
o = st.number_input('Over No.(Not Greater Than Overs Played in 2nd Innings)', min_value=1, max_value=50, value=50)
h = st.text_input('URL(ESPN CRICINFO >Select Match > Click On Overs)',
                  'https://www.espncricinfo.com/series/icc-cricket-world-cup-2023-24-1367856/australia-vs-south-africa-2nd-semi-final-1384438/match-overs-comparison')

if h == 'https://www.espncricinfo.com/series/icc-cricket-world-cup-2023-24-1367856/australia-vs-south-africa-2nd-semi-final-1384438/match-overs-comparison':
    st.write('Enter Your URL')
else:
    r = requests.get(h)
    b = BeautifulSoup(r.text, 'html.parser')
    venue = b.find(class_='ds-flex ds-items-center').text.split(',')[1]
    
    # Initialize lists for data extraction
    list_score = []
    list_wickets = []
    list_over = []
    list_score1 = []
    list_wickets1 = []
    list_over1 = []
    elements = b.find_all(class_='ds-cursor-pointer ds-pt-1')
    for i, element in enumerate(elements):
        if element.text.split('/'):
            if i % 2 != 0:  # Odd indices
                list_score.append(int(element.text.split('/')[0]))
                list_wickets.append(int(element.text.split('/')[1].split('(')[0]))
            else:  # Even indices
                list_over.append(i // 2 + 1)  # Convert to over number
  for i, element in enumerate(elements):
        if element.text.split('/'):
            if i % 2 == 0:  # Odd indices
                list_score1.append(int(element.text.split('/')[0]))
                list_wickets1.append(int(element.text.split('/')[1].split('(')[0]))
            else:  # Even indices
                list_over1.append(i // 2 + 1)  # Convert to over number
       
    # Create DataFrame
    df = pd.DataFrame({
        'score': list_score,
        'wickets': list_wickets,
        'over': list_over[:len(list_score)]  # Ensure matching lengths
    })
 df1= pd.DataFrame({
        'score1': list_score1,
        'wickets1': list_wickets1,
        'over1': list_over[:len(list_score1)]  # Ensure matching lengths
    })

    # Add computed columns
    df['target'] = df['score'].max() + 1  # Example target
    df['crr'] = df['score'] / df['over']
    df['rrr'] = (df['target'] - df['score']) / (50 - df['over'])
    df['balls_left'] = 300 - (df['over'] * 6)
    df['runs'] = df['score'].diff().fillna(0)
    df['last_10'] = df['runs'].rolling(window=10).sum().fillna(0)
    df['wickets_in_over'] = df['wickets'].diff().fillna(0)
    df['last_10_wicket'] = df['wickets_in_over'].rolling(window=10).sum().fillna(0)

    # Display the scorecard
    col1, col2 = st.columns([1, 1])
    with col1:
        st.write(f"**Bowling Team**: Example Team")
        st.write(f"**Batting Team**: Example Team")
    with col2:
        st.write(f"Target: {df['target'].max()}")
        st.write(f"CRR: {df.iloc[-1]['crr']:.2f}, RRR: {df.iloc[-1]['rrr']:.2f}")
        st.write(f"Runs Left: {df.iloc[-1]['target'] - df.iloc[-1]['score']} in {df.iloc[-1]['balls_left']} balls")
    
    # Display plots
    fig = go.Figure(data=[
        go.Scatter(x=df['over'], y=df['score'], mode='lines+markers', name="Score Progression", line_color='green')
    ])
    fig.update_layout(title='Score Comparison', xaxis_title='Over', yaxis_title='Score')
    st.plotly_chart(fig)

# Handling negative indices in df (This was the source of your error)
if 'score1' in df1.columns:  # Ensure df1 exists before proceeding
    neg_idx = df[df['score1'] < 0].index
    if not neg_idx.empty:
        df = df[:neg_idx[0]]

# Truncate dataframe if `o` is less than 50
lf = df
lf = lf[:int(o)]

# Display sections
selected_section = st.radio("Select Section", ['Score Comparison', 'Win Probability'])
if selected_section == 'Score Comparison':
    st.write(fig)
elif selected_section == 'Win Probability':
    fig3 = go.Figure()
    # Example code for plotting win probability (requires actual logic or data for win/lose columns)
    fig3.add_trace(go.Scatter(x=lf['over'], y=lf['score'], mode='lines', name="Win Probability", line_color='blue', line_width=4))
    fig3.update_layout(title=f"Win Probability of Teams (Target: {df['target'].max()})", height=700)
    st.write(fig3)
