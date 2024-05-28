import streamlit as st
import pickle
import pandas as pd
teams=['Pakistan', 'Zimbabwe', 'Bangladesh', 'South Africa', 'Sri Lanka',
       'West Indies', 'India', 'Afghanistan', 'Australia', 'New Zealand',
       'England', 'Ireland', 'Netherlands', 'Nepal']
venue=['R Premadasa Stadium', 'Sheikh Abu Naser Stadium', 'Hagley Oval',
       'County Ground', 'Shere Bangla National Stadium', 'Eden Gardens',
       'Barsapara Cricket Stadium', 'New Wanderers Stadium',
       'Sheikh Zayed Stadium', 'Kensington Oval', 'Brian Lara Stadium',
       'Harare Sports Club', 'Beausejour Stadium',
       'Himachal Pradesh Cricket Association Stadium',
       'Civil Service Cricket Club',
       'Saurashtra Cricket Association Stadium',
       'Greater Noida Sports Complex Ground', 'Kennington Oval',
       'Melbourne Cricket Ground', 'Bellerive Oval',
       'Sharjah Cricket Stadium', 'Perth Stadium',
       'Central Broward Regional Park Stadium Turf Ground',
       'OUTsurance Oval', 'Warner Park', 'National Cricket Stadium',
       'Sydney Cricket Ground', 'R.Premadasa Stadium',
       'Dubai International Cricket Stadium', 'National Stadium',
       'Providence Stadium',
       'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium',
       'Adelaide Oval', 'Westpac Stadium', 'The Village',
       "Queen's Park Oval", 'Arun Jaitley Stadium', 'Riverside Ground',
       'Sophia Gardens', 'Narendra Modi Stadium', 'Eden Park',
       'Kingsmead', 'Trent Bridge', 'Rajiv Gandhi International Stadium',
       'Vidarbha Cricket Association Stadium', 'Sabina Park',
       'Punjab Cricket Association IS Bindra Stadium',
       'Tribhuvan University International Cricket Ground',
       'Windsor Park', 'Queens Sports Club', 'SuperSport Park',
       'Sir Vivian Richards Stadium', 'AMI Stadium', 'Boland Park',
       'Buffalo Park', 'Saxton Oval',
       'Sylhet International Cricket Stadium', 'University Oval',
       'Headingley', 'Maharashtra Cricket Association Stadium',
       'Western Australia Cricket Association Ground']
venue.sort()
teams.sort()
pipe=pickle.load(open("pipe.pkl","rb"))
st.title("T20 WIN PREDICTOR")
col1,col2=st.columns(2)
with col1:
    batting_team=st.selectbox("Select the Batting team",teams)
with col2:
    bowling_team=st.selectbox("Select the Bowling team",teams)

selected_stadium=st.selectbox("Select Venue",venue)
target=st.number_input("Target",step=1,min_value=0)
col3,col4,col5=st.columns(3)
with col3:
    score=st.number_input('Score',step=1)
with col4:
    overs=st.number_input('Overs Completed',step=1)
with col5:
    wickets=st.number_input('Wickets Out',step=1)
if st.button("Predict Probability"):
    runs_left=target-score
    balls_left=120-(overs*6)
    wickets=10-wickets
    crr=score/overs
    rrr=(runs_left/balls_left)*6
    input_df=pd.DataFrame({'Venue':[selected_stadium], 'Bowling team':[bowling_team], 'Batting team':[batting_team], 'Target Score':[target], 'Runs to Get':[runs_left],
       'Balls Remaining':[balls_left], 'Wickets_Rem':[wickets], 'current runs':[score], 'crr':[crr], 'rrr':[rrr]})
    st.table(input_df)
    result=pipe.predict_proba(input_df)
    loss=result[0][0]
    win=result[0][1]
    st.text("Win percentage of " + batting_team + "- " + str(round(win*100)) + "%")
    st.text("Win percentage of " + bowling_team + "- " + str(round(loss*100)) + "%")

    