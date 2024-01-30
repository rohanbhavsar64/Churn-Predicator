import numpy as np
import pandas as pd
import streamlit as st
df=pd.read_csv('churn.csv')
#st.write(df.head())
st.sidebar.title('PROJECTS')
t=st.sidebar.radio('Parts',['Anaylsis','Predication'])
if t=='Anaylsis':
    st.title('TeleCo-Company Custmor Churn Anylsis')
    a=['SeniorCitizen','Partner','Dependents','MultipleLines','InternetService','OnlineSecurity','DeviceProtection','Contract','PaperlessBilling','PaymentMethod']
    b=['tenure','MonthlyCharges','TotalCharges']
    st.sidebar.title('Visualization')
    c=st.sidebar.selectbox('FOR CATEGORICAL COLUMNS',a)
    d=pd.crosstab(df[c],df['Churn'])
    e=st.sidebar.selectbox('FOR NUMERIC COLUMNS',b)
    col1,col2=st.columns(2)
    with col1:
        st.bar_chart(d)
        with col2:
            st.bar_chart(data=df,x='Churn',y=e)
if t=='Predication':
    st.title('TeleCo-Company Custmor Churn Predication')
    a=['Yes','No']
    col1,col2=st.columns(2)
    with col1:
        SeniorCitizen=st.selectbox('Senior Citizen',[0,1])
    with col2:
        Partner=st.selectbox('Partner',a)
    col1,col3=st.columns(2)
    with col1:
        tenure=st.number_input('tenure')
    with col3:
        MultipleLines=st.selectbox('MultipleLines',a)
    b=df['InternetService'].unique()
    col1,col2,col3=st.columns(3)
    with col1:
        InternetService=st.selectbox('InternetService',b)
    with col2:
        OnlineSecurity=st.selectbox('OnlineSecurity',a)
    with col3:
        DeviceProtection=st.selectbox('DeviceProtection',a)
    c=df['Contract'].unique()
    col2,col3=st.columns(2)
    with col2:
        Contract=st.selectbox('Contract',c)
    with col3:
        PaperlessBilling=st.selectbox('PaperlessBilling',a)
    d=df['PaymentMethod'].unique()
    col1,col2,col3=st.columns(3)
    with col1:
        PaymentMethod=st.selectbox('PaymentMethod',d)
    with col2:
        MonthlyCharges=st.number_input('MonthlyCharges')
    with col3:
        TotalCharges=st.number_input('TotalCharges')
    X=df[['SeniorCitizen','Partner','tenure','MultipleLines','InternetService','OnlineSecurity','DeviceProtection','Contract','PaperlessBilling','PaymentMethod','MonthlyCharges','TotalCharges']]
    y=df['Churn']
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=40)
    a=['Partner','MultipleLines','InternetService','OnlineSecurity','DeviceProtection','Contract','PaperlessBilling','PaymentMethod']
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    ohe=OneHotEncoder()
    ohe.fit([a])
    trf=ColumnTransformer([
         ('trf',OneHotEncoder(max_categories=12,sparse_output=False),a)
     ]
         ,remainder='passthrough')
    from sklearn.pipeline import Pipeline
    from sklearn.tree import DecisionTreeClassifier
    pipe=Pipeline(steps=[
    ('step1',trf),
    ('step2',DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=4, min_samples_split=2, min_samples_leaf=20, min_weight_fraction_leaf=0.0, max_features=None, random_state=42))
    ]
     )
    pipe.fit(X_train,y_train)
    y_pred=pipe.predict(X_test)
    from sklearn.metrics import confusion_matrix
#confusion_matrix(y_test,y_pred)
    n=pipe.predict_proba(pd.DataFrame(columns=['SeniorCitizen','Partner','tenure','MultipleLines','InternetService','OnlineSecurity','DeviceProtection','Contract','PaperlessBilling','PaymentMethod','MonthlyCharges','TotalCharges'],data=np.array([SeniorCitizen,Partner,tenure,MultipleLines,InternetService,OnlineSecurity,DeviceProtection,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges]).reshape(1,12))).astype(float)
    from sklearn.metrics import confusion_matrix
    st.write(confusion_matrix(y_test,y_pred))
    if st.button('Predict'):
         st.subheader('Probability of Churn: '+str(int(n[0][1]*100))+' %')