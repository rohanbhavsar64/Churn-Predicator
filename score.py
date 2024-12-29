import pandas as pd
from bs4 import BeautifulSoup
import requests
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pickle

# Load flag data
sf = pd.read_csv('flags_iso.csv')

st.header('ODI MATCH ANALYSIS')
st.sidebar.header('Analysis')

selected_section = st.sidebar.radio('Select a Section:', ('Score Comparison', 'Innings Progression', 'Win Probability', 'Current Predictor'))

# Score Comparison
o = st.number_input('Over No. (Not Greater Than Overs Played in 2nd Innings)', value=50)
h = st.text_input('URL (ESPN CRICINFO >Select Match > Click On Overs)', value='https://www.espncricinfo.com/series/icc-cricket-world-cup-2023-24-1367856/australia-vs-south-africa-2nd-semi-final-1384438/match-overs-comparison')

if h == 'https://www.espncricinfo.com/series/icc-cricket-world-cup-2023-24-1367856/australia-vs-south-africa-2nd-semi-final-1384438/match-overs-comparison':
    st.write('Enter Your URL')

url2 = h.replace('match-overs-comparison', 'live-cricket-score')
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
max_len = max(len(list5), len(list4), len(list6), len(list), len(list1), len(list2), len(list3))

list5.extend([None] * (max_len - len(list5)))
list4.extend([None] * (max_len - len(list4)))
list6.extend([None] * (max_len - len(list6)))
list.extend([None] * (max_len - len(list)))
list1.extend([None] * (max_len - len(list1)))
list2.extend([None] * (max_len - len(list2)))
list3.extend([None] * (max_len - len(list3)))

dict = {'batting_team': list5, 'bowling_team': list4, 'venue': list6, 'score': list, 'wickets': list1, 'over': list2, 'target': list3}
df = pd.DataFrame(dict)

df = pd.DataFrame(dict)

df['score'] = df['score'].astype('int')
df1['inng1'] = df1['inng1'].astype('int')
df1['wickets'] = df1['wickets'].astype('int')
df1['previous_wickets'] = df1['wickets'].shift(1)
df1['previous_wickets'].loc[0] = 0
df1['wic'] = df1['wickets'] - df1['previous_wickets']
df1['over'] = df1['over'].astype('int')
df['over'] = df['over'].astype('int')
df['wickets'] = df['wickets'].astype('int')
df['previous_wickets'] = df['wickets'].shift(1)
df['previous_wickets'].loc[0] = 0
df['wic'] = df['wickets'] - df['previous_wickets']
df['target'] = df['target'].astype('int')
df['runs_left'] = df['target'] - df['score']
df = df[df['score'] < df['target']]
df['crr'] = (df['score'] / df['over'])
df['rrr'] = ((df['target'] - df['score']) / (50 - df['over']))
df['balls_left'] = 300 - (df['over'] * 6)
df['runs'] = df['score'].diff()
df['last_10'] = df['runs'].rolling(window=10).sum()
df['wickets_in_over'] = df['wickets'].diff()
df['last_10_wicket'] = df['wickets_in_over'].rolling(window=10).sum()
df = df.fillna(50)
df['match_id'] = 100001

neg_idx = df1[df1['inng1'] < 0].diff().index
if not neg_idx.empty:
    df1 = df1[:neg_idx[0]]
lf = df
lf = lf[:int(o)]

st.subheader('Scorecard')

o = int(o)
if o != 50:
    col1, col2 = st.columns([1, 1])

    with col1:
        bowling_team = df['bowling_team'].unique()[0]
        batting_team = df['batting_team'].unique()[0]

        # Get the URL for the bowling team
        bowling_team_url = sf[sf['Country'] == bowling_team]['URL']
        if not bowling_team_url.empty:
            # Display the bowling team flag and name in the same line
            col_bowling, col_bowling_name = st.columns([1, 3])  # Adjust proportions as needed
            with col_bowling:
                st.image(bowling_team_url.values[0], width=50)  # Adjust width as needed
            with col_bowling_name:
                st.write(f"**{bowling_team}**")

        # Get the URL for the batting team
        batting_team_url = sf[sf['Country'] == batting_team]['URL']
        if not batting_team_url.empty:
            # Display the batting team flag and name in the same line
            col_batting, col_batting_name = st.columns([1, 3])  # Adjust proportions as needed
            with col_batting:
                st.image(batting_team_url.values[0], width=50)  # Adjust width as needed
            with col_batting_name:
                st.write(f"**{batting_team}**")

    with col2:
        st.markdown("<div style='text-align: right;'>", unsafe_allow_html=True)  # Ensure left alignment
        st.write(str(df['target'].unique()[0]) + '/' + str(df1.iloc[-1, 2]))
        st.write('(' + str(df.iloc[o - 1, 5]) + '/' + '50)' + '    ' + str(df.iloc[o - 1, 3]) + '/' + str(df.iloc[o - 1, 4]))
        st.text('crr : ' + str(df.iloc[o - 1, 8].round(2)) + '  rrr : ' + str(df.iloc[o - 1, 9].round(2)))
        st.write(batting_team + ' Required ' + str(df.iloc[o - 1, 7]) + ' runs in ' + str(df.iloc[o - 1, 10]) + ' balls')
        st.markdown("</div>", unsafe_allow_html=True)

else:
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"{df['bowling_team'].unique()[0]}")
        st.write(f"{df['batting_team'].unique()[0]}")

    with col2:
        st.write(str(df['target'].unique()[0]))
        st.write('(' + str(df.iloc[-1, 5]) + '/' + '50) ' + str(df.iloc[-1, 3]) + '/' + str(df.iloc[-1, 4]))

if 'winner' in df.columns and not df['winner'].empty:
    winner = df['winner'].unique()
    if len(winner) > 0:
        st.write(winner[0] + ' Won')
    else:
        st.write("Winner information not available.")

# Plotting Score Comparison
fig = go.Figure(data=[
    go.Scatter(x=df1['over'], y=df1['inng1'], line_width=3, line_color='red', name=df['bowling_team'].unique()[0]),
    go.Scatter(x=lf['over'], y=lf['score'], line_width=3, line_color='green', name=df['batting_team'].unique()[0])
])

wicket_text = lf['wic'].astype(str)
wicket_y = lf['score'] + lf['wic']  # Adjust y-position based on wickets
wicket_y[wicket_y == lf['score']] = None  # Hide scatter points for 0 wickets

wicket_text1 = df1['wic'].astype(str)
wicket_y1 = df1['inng1'] + df1['wic']  # Adjust y-position based on wickets
wicket_y1[wicket_y1 == df1['inng1']] = None  # Hide scatter points for 0 wickets

fig.add_trace(go.Scatter(x=lf['over'], y=wicket_y, mode='markers', name='Wickets', marker_color='green', marker_size=8, text=wicket_text, textposition='top center'))
fig.add_trace(go.Scatter(x=df1['over'], y=wicket_y1, mode='markers', name='Wickets', marker_color='red', marker_size=8, text=wicket_text1, textposition='top center'))

fig.update_layout(title='Score Comparison', xaxis_title='Over', yaxis_title='Score')
import streamlit as st
import plotly.graph_objects as go
import numpy as np

# Assuming these are defined or passed in elsewhere
o = 50  # Example value for o (change this as per your requirement)
  # Replace with actual prediction pipeline
gf = df  # Replace with actual data
with open('pipeline.pkl', 'rb') as file:
    pipe = pickle.load(file)
# Check if 'o' is not equal to 50
gf['match_id']=100001
if o != 50:
    # Execute the logic only if o != 50

    # match_progression function
    def match_progression(x_df, match_id, pipe):
        match = x_df[x_df['match_id'] == match_id]
        match = match[(match['balls_left'] % 6 == 0)]
        temp_df = match[['batting_team', 'bowling_team', 'venue', 'score', 'wickets', 'runs_left', 'balls_left', 'crr', 'rrr', 'last_10', 'last_10_wicket']].fillna(0)
        temp_df = temp_df[temp_df['balls_left'] != 0]
        
        if temp_df.empty:
            print("Error: Match is not Existed")
            return None, None
        
        result = pipe.predict_proba(temp_df)
        temp_df['lose'] = np.round(result.T[0] * 100, 1)
        temp_df['win'] = np.round(result.T[1] * 100, 1)
        temp_df['end_of_over'] = (300 - temp_df['balls_left']) / 6
        target = (temp_df['score'] + temp_df['runs_left'] + 1).values[0]
        
        # Handle runs
        runs = temp_df['runs_left'].tolist()
        new_runs = runs[:]
        new_runs.insert(0, target)
        temp_df['runs_after_over'] = np.array(runs)[:-1] - np.array(new_runs)
        
        # Handle wickets
        wickets = (10 - temp_df['wickets']).tolist()
        new_wickets = wickets[:]
        new_wickets.insert(0, 10)
        wickets.append(0)
        w = np.array(wickets)
        nw = np.array(new_wickets)
        temp_df['wickets_in_over'] = (nw - w)[0:temp_df.shape[0]]
        temp_df['wickets'] = temp_df['wickets_in_over'].cumsum()

        # Add remaining columns
        temp_df['batting_team'] = match['batting_team']
        temp_df['bowling_team'] = match['bowling_team']
        temp_df['venue'] = match['venue']
        temp_df['score'] = match['score']
        
        print("Target-", target)
        
        # Return the processed dataframe and target score
        temp_df = temp_df[['batting_team', 'bowling_team', 'end_of_over', 'runs_after_over', 'wickets_in_over', 'score', 'wickets', 'lose', 'win', 'venue']]
        return temp_df, target

    # Call match_progression with parameters
    temp_df, target = match_progression(gf, 100001, pipe)
    st.write(temp_df)
    # Filter nd clean data
    temp_df = temp_df[temp_df['runs_after_over'] >= 0]
    temp_df = temp_df[temp_df['wickets_in_over'] >= 0]

    # Plotting the progressions using plotly
    fig2 = go.Figure()

    # Plot runs after over
    runs = fig2.add_trace(go.Bar(x=temp_df['end_of_over'], y=temp_df['runs_after_over'], name='Runs in Over', marker_color='purple'))

    # Plot wickets in over
    wicket_text = temp_df['wickets_in_over'].astype(str)
    wicket_y = temp_df['runs_after_over'] + temp_df['wickets_in_over']
    wicket_y[wicket_y == temp_df['runs_after_over']] = None  # Hide for zero wickets
    wicket = fig2.add_trace(go.Scatter(x=temp_df['end_of_over'], y=wicket_y, mode='markers', name='Wickets in Over', marker_color='orange', marker_size=11, text=wicket_text, textposition='top center'))

    # Set layout
    fig2.update_layout(title='Innings Progression')

    # Create win probability graph
    fig3 = go.Figure()
    batting_team = fig3.add_trace(go.Scatter(x=temp_df.iloc[10:, :]['end_of_over'], y=temp_df.iloc[10:, :]['win'], mode='lines', name=temp_df['batting_team'].unique()[0], line_color='green', line_width=4))
    bowling_team = fig3.add_trace(go.Scatter(x=temp_df.iloc[10:, :]['end_of_over'], y=temp_df.iloc[10:, :]['lose'], mode='lines', name=temp_df['bowling_team'].unique()[0], line_color='red', line_width=4))
    
    # Set layout for win probability chart
    fig3.update_layout(title='Win Probability Of Teams : Target-' + str(target), height=700)

    # Current prediction pie chart
    tf = gf[['batting_team', 'bowling_team', 'venue', 'score', 'wickets', 'runs_left', 'balls_left', 'crr', 'rrr', 'last_10', 'last_10_wicket']]
    n = pipe.predict_proba(pd.DataFrame(columns=['batting_team', 'bowling_team', 'venue', 'score', 'wickets', 'runs_left', 'balls_left', 'crr', 'rrr', 'last_10', 'last_10_wicket'], data=np.array(tf.iloc[-1, :]).reshape(1, 11))).astype(float)
    probablity1 = int(n[0][1] * 100)
    probablity2 = int(n[0][0] * 100)
    data = [probablity1, probablity2]
    data1 = [temp_df['bowling_team'].unique()[0], temp_df['batting_team'].unique()[0]]

    fig4 = go.Figure(data=[go.Pie(labels=data1, values=data, hole=.5)])
    fig4.update_layout(title='Current Predictor')

    # Display the figures in streamlit
    if selected_section == 'Score Comparison':
        st.write(fig)
    elif selected_section == 'Session Distribution':
        st.write(fig1)
    elif selected_section == 'Innings Progression':
        st.write(fig2)
    elif selected_section == 'Win Probability':
        st.write(fig3)
    elif selected_section == 'Current Predictor':
        st.write(fig4)

else:
    # Handle match over case
    st.write('Match Over')

# Display different sections based on the sidebar selection
if selected_section == 'Score Comparison':
    st.write(fig)
elif selected_section == 'Innings Progression':
    # Assuming fig1 is defined in your code for innings progression
    st.write(fig1)
elif selected_section == 'Win Probability':
    # Assuming fig3 is defined in your code for win probability
    st.write(fig3)
elif selected_section == 'Current Predictor':
    if o != 50:
        st.write(fig4)  # Show current prediction pie chart
    else:
        st.write('Match Over')
