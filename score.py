import pandas as pd 
from bs4 import BeautifulSoup
import requests 
import streamlit as st 
import numpy as np
sf=pd.read_csv('flags_iso.csv')
st.header('ODI MATCH ANALYSIS')
st.sidebar.header('Analysis')
selected_section = st.sidebar.radio('Select a Section:', 
                                     ('Score Comparison', 'Session Distribution', 'Innings Progression', 'Win Probability','Current Predictor'))

# Define the function for Score Comparison
o=st.number_input('Over No.(Not Greater Than Overs Played in 2nd Innings)') or 50
h = st.text_input('URL( ESPN CRICINFO >Select Match > Click On Overs )') or 'https://www.espncricinfo.com/series/icc-cricket-world-cup-2023-24-1367856/australia-vs-south-africa-2nd-semi-final-1384438/match-overs-comparison'
if (h=='https://www.espncricinfo.com/series/icc-cricket-world-cup-2023-24-1367856/australia-vs-south-africa-2nd-semi-final-1384438/match-overs-comparison'):
    st.write('Enter Your URL')

url2 = h.replace('match-overs-comparison', 'live-cricket-score')
r = requests.get(h)
#r1=requests.get('https://www.espncricinfo.com/series/icc-cricket-world-cup-2023-24-1367856/india-vs-new-zealand-1st-semi-final-1384437/full-scorecard')
b=BeautifulSoup(r.text,'html')
venue=b.find(class_='ds-flex ds-items-center').text.split(',')[1]
list=[]
list1=[]
list2=[]
list3=[]
list4=[]
list5=[]
list6=[]
list7=[]
list8=[]
list9=[]
list10=[]
#print(b.find_all(class_='ds-text-tight-s ds-font-regular ds-flex ds-justify-center ds-items-center ds-w-7 ds-h-7 ds-rounded-full ds-border ds-border-ui-stroke ds-bg-fill-content-prime')[49].text)
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
            list9.append(i/2+1)
            list10.append(element.text.split('/')[1].split('(')[0])
            
dict1={'inng1':list8,'over':list9,'wickets':list10}
df1=pd.DataFrame(dict1)
for i in range(len(list)):
    list2.append(b.find_all(class_='ds-text-tight-s ds-font-regular ds-flex ds-justify-center ds-items-center ds-w-7 ds-h-7 ds-rounded-full ds-border ds-border-ui-stroke ds-bg-fill-content-prime')[i].text)
    list3.append(b.find(class_='ds-text-compact-m ds-text-typo ds-text-right ds-whitespace-nowrap').text.split('/')[0])
    list4.append(b.find_all('th',class_='ds-min-w-max')[1].text)
    list5.append(b.find_all('th',class_='ds-min-w-max')[2].text)
    list6.append(b.find(class_='ds-flex ds-items-center').text.split(',')[1])
    if o==50:
        list7.append(b.find(class_='ds-text-tight-s ds-font-medium ds-truncate ds-text-typo').text.split(' ')[0])
if o==50:
    dict = {'batting_team': list5, 'bowling_team': list4,'venue':list6,'score':list,'wickets':list1,'over':list2,'target':list3,'winner':list7} 
else:
    dict = {'batting_team': list5, 'bowling_team': list4,'venue':list6,'score':list,'wickets':list1,'over':list2,'target':list3} 
df=pd.DataFrame(dict)

df['score']=df['score'].astype('int')
df1['inng1']=df1['inng1'].astype('int')
df1['wickets']=df1['wickets'].astype('int')
df1['previous_wickets'] = df1['wickets'].shift(1)
df1['previous_wickets'].loc[0]=0
df1['wic']=df1['wickets']-df1['previous_wickets']
df1['over']=df1['over'].astype('int')
df['over']=df['over'].astype('int')
df['wickets']=df['wickets'].astype('int')
df['previous_wickets'] = df['wickets'].shift(1)
df['previous_wickets'].loc[0]=0
df['wic']=df['wickets']-df['previous_wickets']
df['target']=df['target'].astype('int')
df['runs_left']=df['target']-df['score']
df=df[df['score']<df['target']]
df['crr']=(df['score']/df['over'])
df['rrr']=((df['target']-df['score'])/(50-df['over']))
df['balls_left']=300-(df['over']*6)
df['runs'] = df['score'].diff()
df['last_10']=df['runs'].rolling(window=10).sum()
df['wickets_in_over'] = df['wickets'].diff()
df['last_10_wicket']=df['wickets_in_over'].rolling(window=10).sum()
df=df.fillna(50)
#st.write(df)
df['match_id']=100001
neg_idx = df1[df1['inng1']<0].diff().index
if not neg_idx.empty:
    df1 = df1[:neg_idx[0]]
lf=df
lf=lf[:int(o)]
st.subheader('Scorecard')
o=int(o)
if o != 50:
    # Create a single row with two columns
    col1, col2 = st.columns([1, 1])  # Equal width columns

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
        # Adjust the layout of col2 to be left-aligned
        st.markdown("<div style='text-align: right;'>", unsafe_allow_html=True)  # Ensure left alignment
        st.write(str(df['target'].unique()[0]) + '/' + str(df1.iloc[-1, 2]))
        st.write('(' + str(df.iloc[o - 1, 5]) + '/' + '50)' + '    ' + str(df.iloc[o - 1, 3]) + '/' + str(df.iloc[o - 1, 4]))
        st.text('crr : ' + str(df.iloc[o - 1, 8].round(2)) + '  rrr : ' + str(df.iloc[o - 1, 9].round(2)))
        st.write(batting_team + ' Required ' + str(df.iloc[o - 1, 7]) + ' runs in ' + str(df.iloc[o - 1, 10]) + ' balls')
        st.markdown("</div>", unsafe_allow_html=True)  # Close the div for left alignment

    # Display teams and results
else:
  col1, col2 = st.columns(2)
  with col1:
    st.write(f"**{df['bowling_team'].unique()[0]}**")
    st.write(f"**{df['batting_team'].unique()[0]}**")
  with col2:
    st.write(str(df['target'].unique()[0]))
    st.write('(' + str(df.iloc[-1, 5]) + '/' + '50)   ' + str(df.iloc[-1, 3]) + '/' + str(df.iloc[-1, 4]))

  if 'winner' in df.columns and not df['winner'].empty:
    winner = df['winner'].unique()
    if len(winner) > 0:
      st.write(winner[0] + ' Won')
    else:
      st.write("Winner information not available.")
import plotly.graph_objects as go
fig = go.Figure(data=[
    go.Scatter(x=df1['over'], y=df1['inng1'],line_width=3,line_color='red',name=df['bowling_team'].unique()[0]),
    go.Scatter(x=lf['over'], y=lf['score'],line_width=3,line_color='green',name=df['batting_team'].unique()[0])
])
wicket_text = lf['wic'].astype(str)
wicket_y =lf['score']+lf['wic'] # adjust y-position based on wickets
wicket_y[wicket_y == lf['score']] = None  # hide scatter points for 0 wickets
wicket_text1 = df1['wic'].astype(str)
wicket_y1 =df1['inng1']+df1['wic'] # adjust y-position based on wickets
wicket_y1[wicket_y1 == df1['inng1']] = None  # hide scatter points for 0 wickets
wicket = fig.add_trace(go.Scatter(x=lf['over'], y=wicket_y,  # use adjusted y-position
                                  mode='markers',name='Wickets',
                                  marker_color='green',marker_size=8,
                                  text=wicket_text, textposition='top center'))
wicket = fig.add_trace(go.Scatter(x=df1['over'], y=wicket_y1,  # use adjusted y-position
                                  mode='markers',name='Wickets',
                                  marker_color='red',marker_size=8,
                                  text=wicket_text1, textposition='top center'))
fig.update_layout(title='Score Comperison',
                  xaxis_title='Over',
                  yaxis_title='Score')
gf=df

import pickle
with open('pipeline.pkl', 'rb') as file:
    pipe = pickle.load(file)
predictions = pipe.predict_proba(pd.DataFrame(columns=['batting_team','bowling_team','venue','score','wickets','runs_left','balls_left','crr','rrr','last_10','last_10_wicket'],data=np.array(['India','Australia','Punjab Cricket Association Stadium, Mohali',185,4,93,108,5.6,5.41,42.0,3.0]).reshape(1,11)))

# Now you can use the loaded pipeline to make prediction
if int(o)!=50:
    gf = gf[:int(o)]
neg_idx = gf[gf['runs'] < 0].index
if not neg_idx.empty:
    gf = gf[:neg_idx[0]]
neg_idx = gf[gf['wickets']==10].index
if not neg_idx.empty:
    gf = gf[:neg_idx[0]]
    
import streamlit as st

def match_progression(x_df,match_id,pipe):
    match = x_df[x_df['match_id'] == match_id]
    match = match[(match['balls_left']%6 == 0)]
    temp_df = match[['batting_team','bowling_team','venue','score','wickets','runs_left','balls_left','crr','rrr','last_10','last_10_wicket']].fillna(0)
    temp_df = temp_df[temp_df['balls_left'] != 0]
    if temp_df.empty:
        print("Error: Match is not Existed")
        return None, None
    result = pipe.predict_proba(temp_df)
    temp_df['lose'] = np.round(result.T[0]*100,1)
    temp_df['win'] = np.round(result.T[1]*100,1)
    temp_df['end_of_over'] = (300-temp_df['balls_left'])/6    
    target = (temp_df['score']+temp_df['runs_left']+1).values[0]
    runs = temp_df['runs_left'].tolist()
    new_runs = runs[:]
    runs.insert(0,target)
    temp_df['runs_after_over'] = np.array(runs)[:-1] - np.array(new_runs)
    temp_df['batting_team']=match['batting_team']
    temp_df['bowling_team']=match['bowling_team']
    wickets = (10 - temp_df['wickets']).tolist()
    new_wickets = wickets[:]
    new_wickets.insert(0,10)
    wickets.append(0)
    w = np.array(wickets)
    nw = np.array(new_wickets)
    temp_df['wickets_in_over'] = (nw-w)[0:temp_df.shape[0]]
    temp_df['wickets']=temp_df['wickets_in_over'].cumsum()
    temp_df['venue']=match['venue']
    temp_df['score']=match['score']
    print("Target-",target)
    temp_df = temp_df[['batting_team','bowling_team','end_of_over','runs_after_over','wickets_in_over','score','wickets','lose','win','venue']]
    return temp_df,target

temp_df, target = match_progression(gf,100001, pipe)
temp_df=temp_df[temp_df['runs_after_over']>=0]
temp_df = temp_df[temp_df['wickets_in_over'] >= 0]
import plotly.graph_objects as go
import plotly.express as px
        #fig = go.Figure()
        #runs = fig.add_trace(go.Bar(x=temp_df['end_of_over'], y=temp_df['runs_after_over'], name='Runs in Over',marker=dict(color='purple')))
        #wicket_text = temp_df['wickets_in_over'].astype(str)
        #wicket_y = temp_df['runs_after_over'] + temp_df['wickets_in_over'] * 0.6  # adjust y-position based on wickets
        #wicket_y[wicket_y == temp_df['runs_after_over']] = None  # hide scatter points for 0 wickets

        #wicket = fig.add_trace(go.Scatter(x=temp_df['end_of_over'], y=wicket_y,  # use adjusted y-position
                                          #  mode='markers', name='Wickets in Over',
                                           # marker=dict(color='orange', size=10),
                                            #text=wicket_text, textposition='top center'))

# Line plots for batting and bowling teams
        
        #fig.update_layout(
          #  title='Target-' + str(target),
         #   width=800,  # Set the width of the chart
        #    height=700  # Set the height of the chart
       # )
        #fig.update_layout(title='Target-' + str(target))
       # st.write(fig) 

#st.write(temp_df)
fig1 = go.Figure()
        

# Determine which team has the upper hand
        

# Line chart for the team with the upper hand
# Calculate the midpoint of the y-axis
midpoint = 50
a2=gf['bowling_team'].unique()[0]
b2=gf['batting_team'].unique()[0]

# Line chart for batting and bowling teams
import plotly.graph_objects as go
import plotly.express as px
fig2=go.Figure()
runs = fig2.add_trace(go.Bar(x=temp_df['end_of_over'], y=temp_df['runs_after_over'], name='Runs in Over',marker_color='purple'))
wicket_text = temp_df['wickets_in_over'].astype(str)
wicket_y = temp_df['runs_after_over']+temp_df['wickets_in_over']  # adjust y-position based on wickets
wicket_y[wicket_y == temp_df['runs_after_over']] = None  # hide scatter points for 0 wickets
wicket = fig2.add_trace(go.Scatter(x=temp_df['end_of_over'], y=wicket_y,  # use adjusted y-position
                                  mode='markers', name='Wickets in Over',
                                  marker_color='orange',marker_size=11,
                                  text=wicket_text, textposition='top center'))
fig2.update_layout(title='Innings Progression')
fig3 = go.Figure()
batting_team = fig3.add_trace(go.Scatter(x=temp_df.iloc[10:,:]['end_of_over'], y=temp_df.iloc[10:,:]['win'], mode='lines', name=temp_df['batting_team'].unique()[0],line_color='green', line_width=4))
bowling_team = fig3.add_trace(go.Scatter(x=temp_df.iloc[10:,:]['end_of_over'], y=temp_df.iloc[10:,:]['lose'], mode='lines', name=temp_df['bowling_team'].unique()[0],line_color='red', line_width=4))
fig3.update_layout(
    title='Target-' + str(target),
    height=700  # Set the height of the chart
)
fig3.update_layout(title='Win Probablity Of Teams :Target-' + str(target))


tf=gf[['batting_team','bowling_team','venue','score','wickets','runs_left','balls_left','crr','rrr','last_10','last_10_wicket']]
if o!=50:
    n=pipe.predict_proba(pd.DataFrame(columns=['batting_team','bowling_team','venue','score','wickets','runs_left','balls_left','crr','rrr','last_10','last_10_wicket'],data=np.array(tf.iloc[-1,:]).reshape(1,11))).astype(float)
    probablity1=int(n[0][1]*100)
    probablity2=int(n[0][0]*100)
    data=[probablity1,probablity2]
    data1=[b2,a2]
    import plotly.graph_objects as go
    fig4 = go.Figure(data=[go.Pie(labels=data1, values=data, hole=.5)])
    fig4.update_layout(title='Current Predicator')


if selected_section == 'Score Comparison':
    st.write(fig)
elif selected_section == 'Session Distribution':
    st.write(fig1)
elif selected_section == 'Innings Progression':
    st.write(fig2)
elif selected_section == 'Win Probability':
    st.write(fig3)
elif selected_section == 'Current Predictor':
  if o!=50:  
    st.write(fig4)
  else:
    st.write('Match Over')
    
