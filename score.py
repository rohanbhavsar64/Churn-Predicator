import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

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
