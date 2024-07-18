#!/usr/bin/env python
# coding: utf-8

# In[179]:


import pandas as pd
import numpy as np



# In[181]:


df=pd.read_csv('deliver (3).csv')


# In[182]:


df


# In[272]:


match_df=pd.read_csv('matches (2).csv')


# In[273]:


match_df


# In[183]:


match=pd.read_csv('matches.csv',usecols=['id','venue','winner','toss_winner'])


# In[184]:


match['venue'].unique()


# In[185]:


match.head()


# In[186]:


df=df.merge(match[['venue','winner','toss_winner','id']],left_on='match_id',right_on='id')


# In[187]:


df.head(10)


# In[188]:


df.drop(columns=[
    'wide_runs',
    'batsman_runs',
    'extra_runs',
    'dismissal_kind',
    'fielder'
    
],inplace=True)


# In[189]:


df


# In[190]:


df.drop(columns=[
    'bowler',
    'bye_runs',
    'legbye_runs',
    'noball_runs',
    'id'
],inplace=True)


# In[191]:


df


# In[192]:


df=df[df['inning']==2]


# In[193]:


df['score']=df.groupby('match_id').cumsum()['total']


# In[194]:


df.describe()





# In[196]:


df


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


df


# In[202]:


df['player_dismissed']=df['player_dismissed'].fillna("0")
df


# In[203]:


df['player_dismissed']=list(map(lambda x:x if x== "0" else "1",df['player_dismissed']))


# In[204]:


df['player_dismissed']=df['player_dismissed'].astype(int)


# In[205]:


df


# In[206]:


df['wickets']=df.groupby('match_id').cumsum()['player_dismissed']


# In[207]:


df


# In[208]:


df[df['wickets']>10].shape


# In[209]:


groups = df.groupby('match_id')

match_ids = df['match_id'].unique()
last_ten = []
for id in match_ids:
    last_ten.extend(groups.get_group(id).rolling(window=60).sum()['total'].values.tolist())


# In[210]:


df['last_10']=last_ten


# In[211]:


groups = df.groupby('match_id')

match_ids = df['match_id'].unique()
last_ten = []
for id in match_ids:
    last_ten.extend(groups.get_group(id).rolling(window=60).sum()['player_dismissed'].values.tolist())


# In[212]:


df['last_10_wicket']=last_ten


# In[213]:


df


# In[214]:


df=df.fillna(0)


# In[215]:


df.head()


# In[216]:


df1


# In[217]:


df1=df1[df1['inning']==1]


# In[218]:


df1


# In[219]:


df = df1.groupby('match_id').sum()['total'].reset_index().merge(df,on='match_id')


# In[220]:


df


# In[221]:


df.info()


# In[222]:


df['crr']=(df['score']*6)/(300-df['balls_left'])


# In[223]:


df['runs_left']=df['total_x']-df['score']


# In[224]:


df['rrr']=(df['runs_left']*6)/df['balls_left']


# In[225]:


df=df[df['runs_left']>=0]


# In[226]:


df.info()


# In[227]:


new_df=df[['batting_team','bowling_team','venue','toss_winner','score','wickets','batsman','non_striker','runs_left','balls_left','crr','rrr','last_10','last_10_wicket','winner']]


# In[228]:


new_df


# In[229]:


def result(raw):
    return 1 if(raw['batting_team']==raw['winner']) else 0


# In[230]:


new_df['winner']=new_df.apply(result,axis=1)


# In[231]:


new_df['batsman']=new_df['batsman'].str.split(' ').str.get(-1)
new_df['non_striker']=new_df['non_striker'].str.split(' ').str.get(-1)
new_df


# In[232]:


new_df.info()


# In[233]:


new_df=new_df[new_df['rrr']<4000]


# In[234]:


new_df=new_df.dropna()
new_df


# In[235]:


data_to_add={'batting_team':'India','bowling_team':'Australia','venue':'Brisbane Cricket Ground, Woolloongabba','toss_winner':['India','Australia'],'score':[48,98],'wickets':1,'batsman':['Gill','Kohli'],'non_striker':['Kohli','Gill'],'runs_left':[200,150],'balls_left':[240,180],'crr':[4.8,4.9],'rrr':[5.0,5.0],'last_10':[48.0,50.0],'last_10_wicket':[1.0,0],'winner':[0,1]}
addendom=pd.DataFrame(columns=new_df.columns,data=data_to_add)
addendom


# In[236]:


data_to_add={'batting_team':'India','bowling_team':'Australia','venue':'Brisbane Cricket Ground, Woolloongabba','toss_winner':['India','Australia'],'score':[48,98],'wickets':1,'batsman':['Returaj','Kohli'],'non_striker':['Kohli','Returaj'],'runs_left':[200,150],'balls_left':[240,180],'crr':[4.8,4.9],'rrr':[5.0,5.0],'last_10':[48.0,50.0],'last_10_wicket':[1.0,0],'winner':[0,1]}
addendom1=pd.DataFrame(columns=new_df.columns,data=data_to_add)
addendom1


# In[237]:


new_df=pd.concat([new_df,addendom])
new_df=new_df.reset_index(drop=True)
new_df=pd.concat([new_df,addendom1])
new_df=new_df.reset_index(drop=True)


# In[238]:


new_df


# In[239]:


new_df.to_csv('win.csv')


# In[240]:


X=new_df.drop(columns='winner')
y=new_df['winner']


# In[241]:


X


# In[242]:


y


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
ohe.fit([['batting_team','bowling_team','venue','toss_winner','batsman','non_striker']])
trf=ColumnTransformer([
    ('trf',OneHotEncoder(max_categories=6,sparse_output=False,handle_unknown = 'ignore'),['batting_team','bowling_team','venue','toss_winner','batsman','non_striker'])
]
,remainder='passthrough')


# In[246]:


ohe.categories_


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


n=pipe.predict_proba(pd.DataFrame(columns=['batting_team','bowling_team','venue','toss_winner','score','wickets','batsman','non_striker','runs_left','balls_left','crr','rrr','last_10','last_10_wicket'],data=np.array(['India','Australia','Punjab Cricket Association Stadium, Mohali','Pakistan',185,4,'Rahul','Pandya',93,108,5.6,5.41,42.0,3.0]).reshape(1,14))).astype(float)


# In[252]:


print("Win Chances of Batting team is:", n[0][1]*100,"%")
print("Win Chances of Bowling team is:", n[0][0]*100,"%")
if(n[0][0]>0.5):
    print("Forcast:Bowlingteam")
else:
        print("Forecast:Battingteam")
x=[n[0][0],n[0][1]]
y=['b1''b2']


# In[253]:


df2=new_df['venue'].value_counts()
df2.to_csv()


# In[254]:


df2=new_df['batsman'].value_counts()
df2.to_csv()


# In[255]:


X.sample(5)


# In[256]:


import pickle


# In[257]:


filename = 'Win Predicator.log'
pickle.dump(pipe, open(filename, 'wb'))


# In[258]:


loaded_model = pickle.load(open('Win Predicator.log', 'rb'))


# In[259]:


loaded_model.predict_proba(pd.DataFrame(columns=['batting_team','bowling_team','venue','toss_winner','score','wickets','batsman','non_striker','runs_left','balls_left','crr','rrr','last_10','last_10_wicket'],data=np.array(['India','Australia','Punjab Cricket Association Stadium, Mohali','Pakistan',185,4,'Rahul','Pandya',93,108,5.6,5.41,42.0,3.0]).reshape(1,14))).astype(float)


# In[260]:


def match_progression(x_df,match_id,pipe):
    match = x_df[x_df['match_id'] == match_id]
    match = match[(match['balls_left']%6 == 0)]
    temp_df = match[['batting_team','bowling_team','venue','toss_winner','score','wickets','batsman','non_striker','runs_left','balls_left','crr','rrr','last_10','last_10_wicket']].fillna(0)
    temp_df = temp_df[temp_df['balls_left'] != 0]
    if temp_df.empty:
        print("Error: Match is not Existed")
        a=1
        return None, None
    result = pipe.predict_proba(temp_df)
    temp_df['lose'] = np.round(result.T[0]*100,1)
    temp_df['win'] = np.round(result.T[1]*100,1)
    temp_df['end_of_over'] = range(1,temp_df.shape[0]+1)
    
    target = (temp_df['score']+temp_df['runs_left']).values[0]
    runs = list(temp_df['runs_left'].values)
    new_runs = runs[:]
    runs.insert(0,target)
    temp_df['runs_after_over'] = np.array(runs)[:-1] - np.array(new_runs)
    wickets = list(temp_df['wickets'].values)
    new_wickets = wickets[:]
    new_wickets.insert(0,10)
    wickets.append(0)
    w = np.array(wickets)
    nw = np.array(new_wickets)
    temp_df['wickets_in_over'] = (w-nw)[0:temp_df.shape[0]]
    
    print("Target-",target)
    temp_df = temp_df[['end_of_over','runs_after_over','wickets_in_over','lose','win']]
    return temp_df,target


# In[270]:


temp_df,target = match_progression(df,72,pipe)
temp_df


# In[271]:


import plotly.graph_objects as go
fig = go.Figure()
wicket=fig.add_trace(go.Scatter(x=temp_df['end_of_over'], y=temp_df['wickets_in_over'], mode='markers',name='Wickets in Over', marker=dict(color='yellow')))
batting_team=fig.add_trace(go.Scatter(x=temp_df['end_of_over'], y=temp_df['win'], mode='lines',name='Batting side', line=dict(color='#00a65a', width=3)))
bowling_team=fig.add_trace(go.Scatter(x=temp_df['end_of_over'], y=temp_df['lose'], mode='lines',name='Bowling Side', line=dict(color='red', width=4)))
runs=fig.add_trace(go.Bar(x=temp_df['end_of_over'], y=temp_df['runs_after_over'],name='Runs in Over'))
fig.update_layout(title='Target-' + str(target))
import streamlit as st
st.write(fig)



