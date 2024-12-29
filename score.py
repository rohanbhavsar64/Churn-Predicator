import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pandas as pd 
from bs4 import BeautifulSoup
import requests 
import streamlit as st 
import numpy as np
import plotly.graph_objects as go
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
    b = BeautifulSoup(r.text, 'html')
    venue = b.find(class_='ds-flex ds-items-center').text.split(',')[1]
    
    # Initialize lists for data extraction
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

    # Data processing
    df['score'] = df['score'].astype('int')
    df1['inng1'] = df1['inng1'].astype('int')
    df1['wickets'] = df1['wickets'].astype('int')
    df1['previous_wickets'] = df1['wickets'].shift(1)
    df1['previous_wickets'].loc[0] = 0  # Corrected this line

# Data preparation and calculations
df['previous_wickets'].loc[0] = 0
df['wic'] = df['wickets'] - df['previous_wickets']
df['target'] = df['target'].astype('int')
df['runs_left'] = df['target'] - df['score']
df = df[df['score'] < df['target']]  # Filter only scores below the target
df['crr'] = df['score'] / df['over']
df['rrr'] = (df['target'] - df['score']) / (50 - df['over'])
df['balls_left'] = 300 - (df['over'] * 6)
df['runs'] = df['score'].diff()
df['last_10'] = df['runs'].rolling(window=10).sum()
df['wickets_in_over'] = df['wickets'].diff()
df['last_10_wicket'] = df['wickets_in_over'].rolling(window=10).sum()
df = df.fillna(50)
df['match_id'] = 100001

# Handling negative indices in df1
neg_idx = df1[df1['inng1'] < 0].index
if not neg_idx.empty:
    df1 = df1[:neg_idx[0]]

# Truncate dataframe if `o` is less than 50
lf = df
lf = lf[:int(o)]

# Display the scorecard
o = int(o)
if o != 50:
    col1, col2 = st.columns([1, 1])  # Equal width columns
    
    with col1:
        bowling_team = df['bowling_team'].unique()[0]
        batting_team = df['batting_team'].unique()[0]

        # Display team information with images
        bowling_team_url = sf[sf['Country'] == bowling_team]['URL']
        if not bowling_team_url.empty:
            col_bowling, col_bowling_name = st.columns([1, 3])
            with col_bowling:
                st.image(bowling_team_url.values[0], width=50)
            with col_bowling_name:
                st.write(f"**{bowling_team}**")
        
        batting_team_url = sf[sf['Country'] == batting_team]['URL']
        if not batting_team_url.empty:
            col_batting, col_batting_name = st.columns([1, 3])
            with col_batting:
                st.image(batting_team_url.values[0], width=50)
            with col_batting_name:
                st.write(f"**{batting_team}**")
    
    with col2:
        st.markdown("<div style='text-align: right;'>", unsafe_allow_html=True)
        st.write(str(df['target'].unique()[0]) + '/' + str(df1.iloc[-1, 2]))
        st.write(f"({df.iloc[o - 1, 5]}/50) {df.iloc[o - 1, 3]}/{df.iloc[o - 1, 4]}")
        st.text(f"CRR: {df.iloc[o - 1, 8].round(2)} RRR: {df.iloc[o - 1, 9].round(2)}")
        st.write(f"{batting_team} requires {df.iloc[o - 1, 7]} runs in {df.iloc[o - 1, 10]} balls")
        st.markdown("</div>", unsafe_allow_html=True)

else:
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"{df['bowling_team'].unique()[0]}")
        st.write(f"{df['batting_team'].unique()[0]}")
    with col2:
        st.write(str(df['target'].unique()[0]))
        st.write(f"({df.iloc[-1, 5]}/50) {df.iloc[-1, 3]}/{df.iloc[-1, 4]}")

# Winner information
if 'winner' in df.columns and not df['winner'].empty:
    winner = df['winner'].unique()
    if len(winner) > 0:
        st.write(f"{winner[0]} won")
    else:
        st.write("Winner information not available.")

# Plotly graphs
fig = go.Figure(data=[
    go.Scatter(x=df1['over'], y=df1['inng1'], line_width=3, line_color='red', name=df['bowling_team'].unique()[0]),
    go.Scatter(x=lf['over'], y=lf['score'], line_width=3, line_color='green', name=df['batting_team'].unique()[0])
])

# Adding wickets to the plot
wicket_text = lf['wic'].astype(str)
wicket_y = lf['score'] + lf['wic']
wicket_y[wicket_y == lf['score']] = None  # Hide points for 0 wickets

fig.add_trace(go.Scatter(x=lf['over'], y=wicket_y, mode='markers', name='Wickets', 
                         marker_color='green', marker_size=8, text=wicket_text, textposition='top center'))

fig.update_layout(title='Score Comparison', xaxis_title='Over', yaxis_title='Score')

# Display sections
selected_section = st.radio("Select Section", ['Score Comparison', 'Win Probability'])
if selected_section == 'Score Comparison':
    st.write(fig)
elif selected_section == 'Win Probability':
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=temp_df.iloc[10:, :]['end_of_over'], y=temp_df.iloc[10:, :]['win'], mode='lines',
                              name=temp_df['batting_team'].unique()[0], line_color='green', line_width=4))
    fig3.add_trace(go.Scatter(x=temp_df.iloc[10:, :]['end_of_over'], y=temp_df.iloc[10:, :]['lose'], mode='lines',
                              name=temp_df['bowling_team'].unique()[0], line_color='red', line_width=4))
    fig3.update_layout(title=f"Win Probability of Teams (Target: {target})", height=700)
    st.write(fig3)
