import pandas as pd 
from bs4 import BeautifulSoup
import requests 
import streamlit as st 
sf=pd.read_csv('flags_iso.csv')
st.header('ODI MATCH ANALYSIS')
st.sidebar.header('Analysis')
selected_section = st.sidebar.radio('Select a Section:', 
                                     ('Score Comparison', 'Session Distribution', 'Innings Progression', 'Win Probability', 'Current Predictor'))

# Define the function for Score Comparison
o=st.number_input('Over No.(Not Greater Than Overs Played in 2nd Innings)') or 50
h = st.text_input('URL( ESPN CRICINFO >Select Match > Click On Overs )') or 'https://www.espncricinfo.com/series/icc-cricket-world-cup-2023-24-1367856/australia-vs-south-africa-2nd-semi-final-1384438/match-overs-comparison'
if (h=='https://www.espncricinfo.com/series/icc-cricket-world-cup-2023-24-1367856/australia-vs-south-africa-2nd-semi-final-1384438/match-overs-comparison'):
    st.write('Enter Your URL')
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
df1['over']=df1['over'].astype('int')
df['over']=df['over'].astype('int')
df['wickets']=df['wickets'].astype('int')
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
fig.update_layout(title='Score Comperison',
                  xaxis_title='Over',
                  yaxis_title='Score')
gf=df
import pandas as pd
import numpy as np

#df3=pd.read_csv('delive.csv')
#df2=pd.read_csv('del.csv')
#df2=df2[(df2['match_id']>=1364) &(df2['match_id']<=1410) ]
df=pd.read_csv('ball.csv')
#df=pd.concat([df,df2])
import streamlit as st
d=df

df1=df


# In[182]:




# In[272]:


match_df=pd.read_csv('matches (2).csv')


# In[273]:




# In[183]:


match=pd.read_csv('matches (2).csv')


# In[184]:



# In[185]:




# In[186]:


df=df.merge(match[['venue','winner','toss_winner','id']],left_on='match_id',right_on='id')


# In[187]:



# In[188]:


df.drop(columns=[
    'wide_runs',
    'batsman_runs',
    'extra_runs',
    'dismissal_kind',
    'fielder'
    
],inplace=True)


# In[189]:




# In[190]:


df.drop(columns=[
    'bowler',
    'bye_runs',
    'legbye_runs',
    'noball_runs',
    'id'
],inplace=True)


# In[191]:





# In[192]:


df=df[df['inning']==2]


# In[193]:


df['score']=df.groupby('match_id')['total'].cumsum()


# In[194]:








# In[196]:




# In[197]:


df['overs']=df['over']-1


# In[198]:


df['balls_left']=300-(6*(df['over']))-df['ball']


# In[199]:


df.head(10)


# In[200]:


df.drop(columns=[
    'over',
    'ball'
],inplace=True)


# In[201]:





# In[202]:


df['player_dismissed']=df['player_dismissed'].fillna("0")


# In[203]:

df['player_dismissed'] = (df['player_dismissed'] != "0").astype(int)


# In[204]:


df['player_dismissed']=df['player_dismissed'].astype(int)


# In[205]:




# In[206]:


df['wickets']=df.groupby('match_id')['player_dismissed'].cumsum()


# In[207]:





# In[208]


# In[209]:


groups = df.groupby('match_id')

match_ids = df['match_id'].unique()
last_ten = []
for id in match_ids:
    last_ten.extend(groups.get_group(id).rolling(window=60)['total'].sum().values.tolist())


# In[210]:


df['last_10']=last_ten


# In[211]:


groups = df.groupby('match_id')

match_ids = df['match_id'].unique()
last_ten = []
for id in match_ids:
    last_ten.extend(groups.get_group(id).rolling(window=60)['player_dismissed'].sum().values.tolist())


# In[212]:


df['last_10_wicket']=last_ten


# In[213]:





# In[214]:


df=df.fillna(0)


# In[215]:





# In[216]:





# In[217]:


df1=df1[df1['inning']==1]


# In[218]:





# In[219]:


df = df1.groupby('match_id').sum()['total'].reset_index().merge(df,on='match_id')





# In[222]:


df['crr']=(df['score']*6)/(300-df['balls_left'])


# In[223]:


df['runs_left']=df['total_x']-df['score']


# In[224]:


df['rrr']=(df['runs_left']*6)/df['balls_left']


# In[225]:


df=df[df['runs_left']>=0]


# In[226]:



# In[227]:

new_df=df[['batting_team','bowling_team','venue','score','wickets','runs_left','balls_left','crr','rrr','last_10','last_10_wicket','winner']]


# In[228]:


# In[229]:


def result(raw):
    return 1 if(raw['batting_team']==raw['winner']) else 0


# In[230]:


new_df['winner']=new_df.apply(result,axis=1)


# In[231]:



# In[232]:




# In[233]:


new_df=new_df[new_df['rrr']<4000]


# In[234]:


new_df=new_df.dropna()



# In[235]:


# In[237]:






# In[239]:


new_df.to_csv('win.csv')


# In[240]:


X=new_df.drop(columns='winner')
y=new_df['winner']


# In[241]:





# In[243]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from  sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB


# In[244]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


# In[245]:


ohe=OneHotEncoder()
ohe.fit([['batting_team','bowling_team','venue']])
trf=ColumnTransformer([
    ('trf',OneHotEncoder(max_categories=6,sparse_output=False,handle_unknown = 'ignore'),['batting_team','bowling_team','venue'])
]
,remainder='passthrough')


# In[246]:



# In[247]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)


# In[248]:


from sklearn.pipeline import Pipeline


# In[249]:


pipe=Pipeline(steps=[
    ('step1',trf),
    ('step2', LogisticRegression())
]
)


# In[250]:


pipe.fit(X_train,y_train)


# In[251]:


n=pipe.predict_proba(pd.DataFrame(columns=['batting_team','bowling_team','venue','score','wickets','runs_left','balls_left','crr','rrr','last_10','last_10_wicket'],data=np.array(['India','Australia','Punjab Cricket Association Stadium, Mohali',185,4,93,108,5.6,5.41,42.0,3.0]).reshape(1,11))).astype(float)
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
wicket_y = temp_df['runs_after_over']+temp_df['wickets_in_over']*0.4  # adjust y-position based on wickets
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
    st.write(fig4)
if o==50:
    def result(raw):
        return 1 if(raw['batting_team']==raw['winner']) else 0
        gf['winner']=gf.apply(result,axis=1)
        if ((h=='https://www.espncricinfo.com/series/icc-cricket-world-cup-2023-24-1367856/australia-vs-south-africa-2nd-semi-final-1384438/match-overs-comparison') or (o!=50)):
            new_df=new_df
        else:
            new_df = pd.concat([new_df, gf[['batting_team','bowling_team','venue','score','wickets','runs_left','balls_left','crr','rrr','last_10','last_10_wicket','winner']]])
new_df.drop_duplicates(inplace=True)
