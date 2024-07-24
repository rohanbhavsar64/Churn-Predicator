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


df['player_dismissed']=list(map(lambda x:x if x== "0" else "1",df['player_dismissed']))


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


new_df=df[['batting_team','bowling_team','venue','toss_winner','score','wickets','batsman','non_striker','runs_left','balls_left','crr','rrr','last_10','last_10_wicket','winner']]


# In[228]:





# In[229]:


def result(raw):
    return 1 if(raw['batting_team']==raw['winner']) else 0


# In[230]:


new_df['winner']=new_df.apply(result,axis=1)


# In[231]:


new_df['batsman']=new_df['batsman'].str.split(' ').str.get(-1)
new_df['non_striker']=new_df['non_striker'].str.split(' ').str.get(-1)



# In[232]:




# In[233]:


new_df=new_df[new_df['rrr']<4000]


# In[234]:


new_df=new_df.dropna()



# In[235]:


data_to_add={'batting_team':'India','bowling_team':'Australia','venue':'Brisbane Cricket Ground, Woolloongabba','toss_winner':['India','Australia'],'score':[48,98],'wickets':1,'batsman':['Gill','Kohli'],'non_striker':['Kohli','Gill'],'runs_left':[200,150],'balls_left':[240,180],'crr':[4.8,4.9],'rrr':[5.0,5.0],'last_10':[48.0,50.0],'last_10_wicket':[1.0,0],'winner':[0,1]}
addendom=pd.DataFrame(columns=new_df.columns,data=data_to_add)



# In[236]:


data_to_add={'batting_team':'India','bowling_team':'Australia','venue':'Brisbane Cricket Ground, Woolloongabba','toss_winner':['India','Australia'],'score':[48,98],'wickets':1,'batsman':['Returaj','Kohli'],'non_striker':['Kohli','Returaj'],'runs_left':[200,150],'balls_left':[240,180],'crr':[4.8,4.9],'rrr':[5.0,5.0],'last_10':[48.0,50.0],'last_10_wicket':[1.0,0],'winner':[0,1]}
addendom1=pd.DataFrame(columns=new_df.columns,data=data_to_add)



# In[237]:


new_df=pd.concat([new_df,addendom])
new_df=new_df.reset_index(drop=True)
new_df=pd.concat([new_df,addendom1])
new_df=new_df.reset_index(drop=True)





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
ohe.fit([['batting_team','bowling_team','venue','toss_winner','batsman','non_striker']])
trf=ColumnTransformer([
    ('trf',OneHotEncoder(max_categories=6,sparse_output=False,handle_unknown = 'ignore'),['batting_team','bowling_team','venue','toss_winner','batsman','non_striker'])
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


n=pipe.predict_proba(pd.DataFrame(columns=['batting_team','bowling_team','venue','toss_winner','score','wickets','batsman','non_striker','runs_left','balls_left','crr','rrr','last_10','last_10_wicket'],data=np.array(['India','Australia','Punjab Cricket Association Stadium, Mohali','Pakistan',185,4,'Rahul','Pandya',93,108,5.6,5.41,42.0,3.0]).reshape(1,14))).astype(float)
part = st.radio(" ", ["Prediction", "Analysis"],horizontal=True)

if part == "Prediction":
    # Prediction part
    st.title('IPL Win Predictor')
    batting=final_df['batting_team'].unique()
    batting=final_df['batsman'].unique()
    non=final_df['non_striker'].unique()
    shar=final_df['venue'].unique()
    wic1=[0,1,2,3,4,5,6,7,8,9,10]
    col1,col2,col3,col4=st.columns(4)
    with col1:
        a = st.selectbox('batting_team',sorted(batting))
    with col2:
        b = st.selectbox('bowling_team',sorted(batting))
    with col3:
        c= st.selectbox('city',sorted(shar))
    with col4:
        u = st.selectbox('botsman',sorted(batsman))
        
    col1,col2,col3,col4=st.columns(4)
    with col1:
        d= int(st.number_input('runs_left'))
    with col2:
        f=st.selectbox('wickets',wic1)
    with col3:
        g=st.number_input('crr')
    with col4:
        v = st.selectbox('Non striker',sorted(non))
    col1,col2,col3,col4=st.columns(4)
    with col1:
        h=st.number_input('Runs in last 10 overs')
    with col2:
        i=st.selectbox('Wickets in last 10 overs',sorted(wic1))
    with col3:
        e= st.number_input('balls left in Inning')
    with col4:
        w= st.selectbox('Toss Winner',sorted(batting_team))
        
    col1,col2=st.columns(2)
    with col1:
        k=st.number_input('Score')
    with col2:
        l=st.number_input('required run rate')

    n=pipe.predict_proba(pd.DataFrame(columns=['batting_team','bowling_team','venue','toss_winner','score','wickets','batsman','non_striker','runs_left','balls_left','crr','rrr','last_10','last_10_wicket],data=np.array([a,b,c,w,k,f,u,v,d,e,g,l,h,i]).reshape(1,14))).astype(float)
    probablity1=int(n[0][1]*100)
    probablity2=int(n[0][0]*100)
    data=[probablity1,probablity2]
    data1=[a,b]
    if a!=b:
        if st.button('Predict'):
          import plotly.graph_objects as go
          fig = go.Figure(data=[go.Pie(labels=data1, values=data, hole=.5)])
          st.write(fig)
match=match[match['date']>='2019-01-01']

# In[254]:
df2=new_df['batsman'].value_counts()
df2.to_csv()
import streamlit as st
st.header('2019 ODI WORLD CUP ANALYSIS')
# In[255]:
batting=df['batting_team'].unique()
col1, col2 = st.columns(2)
with col1:
    a1 = st.selectbox('team1', sorted(batting))
with col2:
    b1 = st.selectbox('team2', sorted(batting))

match = match[match['team2'] == a1]
match = match[match['team1'] == b1]
#match = match[match['date'] == d1]
g = match['id'].unique()
p= match['date'].unique()
f=st.selectbox('Date', p)
match = match[match['date'] == f]
if a1 == b1:
    st.write('Select Different Teams')
else:
    if match.empty:
        st.write('No match data available')
    else:
        l = match['id'].unique()[0]
df=df[(df['match_id']>=194) & (df['match_id']<=238)]

def match_progression(x_df,match_id,pipe):
    match = x_df[x_df['match_id'] == match_id]
    match = match[(match['balls_left']%6 == 0)]
    temp_df = match[['batting_team','bowling_team','venue','toss_winner','score','wickets','batsman','non_striker','runs_left','balls_left','crr','rrr','last_10','last_10_wicket']].fillna(0)
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
    temp_df['batsman']=match['batsman']
    temp_df['non_striker']=match['non_striker']
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
    temp_df['score']=match['score']
    print("Target-",target)
    temp_df = temp_df[['batting_team','bowling_team','end_of_over','runs_after_over','wickets_in_over','batsman','non_striker','score','wickets','lose','win']]
    return temp_df,target


# In[270]:
if l is None:
    st.write('No Match Available')
else:
    temp_df, target = match_progression(df, l, pipe)
    if temp_df is None:
        st.write("Error: Match is not Existed")
    else:
        print(temp_df)



# In[271]:
import plotly.graph_objects as go
a2=df[df['match_id']==l]['batting_team'].unique()
b2=df[df['match_id']==l]['bowling_team'].unique()
if a1 == b1:
    st.write('No match Available')
else:
    if temp_df is None:
        st.write("Error: Match is not Existed")
    else:
        r1 = match[match['id'] == l]['player_of_match'].unique()
        r2 = match[match['id'] == l]['winner'].unique()
        r3 = df[df['match_id'] == l]['venue'].unique()
        r4=df[df['match_id'] == l]['batting_team'].unique()
        r5 = df[df['match_id'] == l]['toss_winner'].unique()
        r6=df[df['match_id'] == l]['bowling_team'].unique()
        data = {'Field': ['Vanue', 'BattingTeam','BowlingTeam', 'Toss Winner', 'POM', 'Winner'], 'Name': [r3[0], r4[0], r6[0], r5[0], r1[0], r2[0]]} 
        fg = pd.DataFrame(data) 
        st.table(fg)

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
                         line=dict(color='orange', width=2)))
        fig1.update_yaxes(range=[0, 100], tickvals=[0, midpoint, 100], ticktext=[a2, '50%',b2])
        runs = fig1.add_trace(go.Bar(x=temp_df['end_of_over'], y=temp_df['runs_after_over'], name='Runs in Over',marker=dict(color='purple')))
        wicket_text = temp_df['wickets_in_over'].astype(str)
        wicket_y = temp_df['runs_after_over'] + temp_df['wickets_in_over'] * 1  # adjust y-position based on wickets
        wicket_y[wicket_y == temp_df['runs_after_over']] = None  # hide scatter points for 0 wickets

        wicket = fig1.add_trace(go.Scatter(x=temp_df['end_of_over'], y=wicket_y,  # use adjusted y-position
                                            mode='markers', name='Wickets in Over',
                                            marker=dict(color='red', size=10),
                                            text=wicket_text, textposition='top center'))


        fig1.update_layout(xaxis_title='End of Over', yaxis_title='Probability')
        fig1.update_layout(title='Target-' + str(target))

        st.write(fig1)
        e=match.groupby('team1')['winner'].value_counts()[0]
        f=match.groupby('team1')['winner'].value_counts()[1]
        df.groupby('venue')['total_x'].mean().astype(int)
# Pie chart for winner distribution
        winner_counts = match.groupby('team1')['winner'].value_counts().reset_index(name='count')
        fig = px.pie(winner_counts, names='team1', values='count', title='Winner Distribution')
        fig.show()

# Bar chart for mean total_x by venue
        mean_total_x = df.groupby('venue')['total_x'].mean().reset_index(name='mean_total_x')
        fig = px.bar(mean_total_x, x='venue', y='mean_total_x', title='Mean Total X by Venue')
        fig.show()
        
        




