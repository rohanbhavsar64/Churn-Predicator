import pandas as pd 
from bs4 import BeautifulSoup
import requests 
import streamlit as st
h=st.text_input('URL')
if h is None:
    r=requests.get('https://www.espncricinfo.com/series/icc-cricket-world-cup-2023-24-1367856/australia-vs-south-africa-2nd-semi-final-1384438/match-overs-comparison')
else:
    r=requests.get(h)
#r1=requests.get('https://www.espncricinfo.com/series/icc-cricket-world-cup-2023-24-1367856/india-vs-new-zealand-1st-semi-final-1384437/full-scorecard')
b=BeautifulSoup(r.text,'html')
bowling_team=b.find_all(class_='ds-text-tight-l ds-font-bold ds-text-typo hover:ds-text-typo-primary ds-block ds-truncate')[0].text
batting_team=b.find_all(class_='ds-text-tight-l ds-font-bold ds-text-typo hover:ds-text-typo-primary ds-block ds-truncate')[1].text
venue=b.find(class_='ds-flex ds-items-center').text.split(',')[1]
list=[]
list1=[]
list2=[]
list3=[]
list4=[]
list5=[]
list6=[]
#print(b.find_all(class_='ds-text-tight-s ds-font-regular ds-flex ds-justify-center ds-items-center ds-w-7 ds-h-7 ds-rounded-full ds-border ds-border-ui-stroke ds-bg-fill-content-prime')[49].text)
elements = b.find_all(class_='ds-cursor-pointer ds-pt-1')

for i, element in enumerate(elements):
    if element.text.split('/') is None:
        print(' ')
    else:
        if i % 2 != 0:
            list.append(element.text.split('/')[0])
            list1.append(element.text.split('/')[1].split('(')[0])

for i in range(len(list)):
    list2.append(b.find_all(class_='ds-text-tight-s ds-font-regular ds-flex ds-justify-center ds-items-center ds-w-7 ds-h-7 ds-rounded-full ds-border ds-border-ui-stroke ds-bg-fill-content-prime')[i].text)
    list3.append(b.find(class_='ds-text-compact-m ds-text-typo ds-text-right ds-whitespace-nowrap').text.split('/')[0])
    list4.append(b.find_all(class_='ds-text-tight-l ds-font-bold ds-text-typo hover:ds-text-typo-primary ds-block ds-truncate')[0].text)
    list5.append(b.find_all(class_='ds-text-tight-l ds-font-bold ds-text-typo hover:ds-text-typo-primary ds-block ds-truncate')[1].text)
    list6.append(b.find(class_='ds-flex ds-items-center').text.split(',')[1])

dict = {'batting_team': list5, 'bowling_team': list4,'venue':list6,'score':list,'wickets':list1,'over':list2,'target':list3} 
df=pd.DataFrame(dict)
df['score']=df['score'].astype('int')
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
df['last_10_wickets']=df['wickets_in_over'].rolling(window=10).sum()

df=df.dropna()
st.write(df)
df['match_id']=1000001
gf=df
import pandas as pd
import numpy as np



# In[181]
#df3=pd.read_csv('delive.csv')
#df2=pd.read_csv('del.csv')
#df2=df2[(df2['match_id']>=1364) &(df2['match_id']<=1410) ]
df=pd.read_csv('deliv.csv')
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

import streamlit as st
st.header('2019 ODI WORLD CUP ANALYSIS')
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
    temp_df['end_of_over'] = range(1,temp_df.shape[0]+1)
    
    target = (temp_df['score']+temp_df['runs_left']+1).values[0]
    runs = list(temp_df['runs_left'].values)
    new_runs = runs[:]
    runs.insert(0,target)
    temp_df['runs_after_over'] = np.array(runs)[:-1] - np.array(new_runs)
    temp_df['batting_team']=match['batting_team']
    temp_df['bowling_team']=match['bowling_team']
    wickets = list(10-temp_df['wickets'].values)
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
import plotly.graph_objects as go
import plotly.express as px
        #fig = go.Figure()
        #runs = fig.add_trace(go.Bar(x=temp_df['end_of_over'], y=temp_df['runs_after_over'], name='Runs in Over',marker=dict(color='purple')))
        #wicket_text = temp_df['wickets_in_over'].astype(str)
        #wicket_y = temp_df['runs_after_over'] + temp_df['wickets_in_over'] * 1  # adjust y-position based on wickets
        #wicket_y[wicket_y == temp_df['runs_after_over']] = None  # hide scatter points for 0 wickets

        #wicket = fig.add_trace(go.Scatter(x=temp_df['end_of_over'], y=wicket_y,  # use adjusted y-position
                                          #  mode='markers', name='Wickets in Over',
                                           # marker=dict(color='orange', size=10),
                                            #text=wicket_text, textposition='top center'))

# Line plots for batting and bowling teams
        #batting_team = fig.add_trace(go.Scatter(x=temp_df['end_of_over'], y=temp_df['win'], mode='lines', name='Batting Team',
                                           # line=dict(color='#00a65a', width=3)))
        #bowling_team = fig.add_trace(go.Scatter(x=temp_df['end_of_over'], y=temp_df['lose'], mode='lines', name='Bowling Team',
                                            #line=dict(color='red', width=4)))
        #fig.update_layout(
          #  title='Target-' + str(target),
         #   width=800,  # Set the width of the chart
        #    height=700  # Set the height of the chart
       # )
        #fig.update_layout(title='Target-' + str(target))
       # st.write(fig) 

        fig1 = go.Figure()
        

# Determine which team has the upper hand
        

# Line chart for the team with the upper hand
# Calculate the midpoint of the y-axis
        midpoint = 50
        a2=temp_df['bowling_team'].unique()
        b2=temp_df['batting_team'].unique()

# Line chart for batting and bowling teams
        fig1.add_trace(go.Scatter(x=temp_df['end_of_over'], y=temp_df['win'], mode='lines',
                         line=dict(color='yellow', width=3),name='Probablity'))
        fig1.update_yaxes(range=[0, 100], tickvals=[0, midpoint, 100], ticktext=[a2, '50%',b2])
        runs = fig1.add_trace(go.Bar(x=temp_df['end_of_over'], y=temp_df['runs_after_over'], name='Runs in Over',marker=dict(color='purple')))
        wicket_text = temp_df['wickets_in_over'].astype(str)
        wicket_y = temp_df['runs_after_over'] + temp_df['wickets_in_over'] * 1  # adjust y-position based on wickets
        wicket_y[wicket_y == temp_df['runs_after_over']] = None  # hide scatter points for 0 wickets

        wicket = fig1.add_trace(go.Scatter(x=temp_df['end_of_over'], y=wicket_y,  # use adjusted y-position
                                            mode='markers', name='Wickets in Over',
                                            marker=dict(color='red', size=10),
                                            text=wicket_text, textposition='top center'))

