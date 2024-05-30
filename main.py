import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
page_element="""
<style>
[data-testid="stAppViewContainer"]{
  background-image: url("https://media.istockphoto.com/id/177427917/photo/close-up-of-red-cricket-ball-and-bat-sitting-on-grass.jpg?s=1024x1024&w=is&k=20&c=prwXq6gId0T2Lalr1SvTVZWgboWo6siTQmW2PJJ2ZkY=");
  background-size: cover;
}
</style>
"""
st.markdown(page_element, unsafe_allow_html=True)
innings_1=pd.read_csv("first_innings.csv")
innings_2=pd.read_csv("second_innings.csv")
df2=pd.read_csv("pure.csv")
if "pipe" not in st.session_state:
    df=pd.read_csv("df1.csv")
    x=df.drop("Result",axis=1)
    y=df["Result"]
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
    trf=ColumnTransformer([("trf",OneHotEncoder(sparse_output=False,drop="first"),["Venue","Bowling team","Batting team"])],remainder="passthrough")
    st.session_state.pipe=Pipeline(steps=[("step1",trf),("step2",LogisticRegression(solver="liblinear"))])
    x_train.drop(columns='Unnamed: 0',axis=1,inplace=True)
    st.session_state.pipe.fit(x_train,y_train)
def innings_progression(match,first,second):
    match_df_2=innings_2[innings_2["Match ID"]==match]
    match_df_1=innings_1[innings_1["Match ID"]==match]
    target=match_df_2.iloc[0]["current score"]
    match_df_2["Current score"]=match_df_2["current score"]-target
    match_df_2.drop(columns=["current score"],inplace=True)
    match_df_2.rename(columns={"Current score":"current score"},inplace=True)
    match_df_2["balls"]=(match_df_2["Over"]-1)*6+match_df_2["Ball"]
    match_df_2.drop(columns=["Over","Ball","Innings"],inplace=True)
    match_df_1["balls"]=(match_df_1["Over"]-1)*6+match_df_1["Ball"]
    match_df_1.drop(columns=["Over","Ball","Innings"],inplace=True)
    wickets_1=match_df_1[match_df_1["Wicket"]==1]
    wickets_2=match_df_2[match_df_2["Wicket"]==1]
    innings_1_runs=match_df_1["current score"].to_list()
    innings_1_wickets_x=wickets_1["balls"].to_list()
    innings_1_wickets_y=wickets_1["current score"].to_list()
    innings_2_wickets_x=wickets_2["balls"].to_list()
    innings_2_wickets_y=wickets_2["current score"].to_list()
    innings_1_balls=match_df_1["balls"].to_list()
    innings_2_runs=match_df_2["current score"].to_list()
    innings_2_balls=match_df_2["balls"].to_list()
    fig=go.Figure()
    fig.add_trace(go.Line(x=innings_1_balls,y=innings_1_runs,name=first))
    fig.add_trace(go.Line(x=innings_2_balls,y=innings_2_runs,name=second))
    fig.add_trace(go.Scatter(x=innings_1_wickets_x,y=innings_1_wickets_y,mode="markers",marker_color="blue",name="first innings wickets"))
    fig.add_trace(go.Scatter(x=innings_2_wickets_x,y=innings_2_wickets_y,mode="markers",marker_color="red",name="second innings wickets"))
    fig.update_layout(width=800,height=400,title="Innings progression",xaxis_title="Balls",yaxis_title="runs")
    return fig
def match_progression(match):
    train=pd.read_csv("model training dataset.csv")
    x=train.iloc[:,:-1]
    y=train.iloc[:,-1]
    x.drop(columns=["Unnamed: 0"],inplace=True)
    ct=ColumnTransformer([("trf",OneHotEncoder(sparse_output=False,drop="first"),['Venue','Bat First', 'Bat Second'])],remainder="passthrough")
    st.session_state.pipe1=Pipeline(steps=[("step 1",ct),("step 2",LogisticRegression(solver="liblinear"))])
    st.session_state.pipe1.fit(x,y)
    match_df_2=innings_2[innings_2["Match ID"]==match]
    target=match_df_2.iloc[0]["current score"]
    match_df_2["Current score"]=match_df_2["current score"]-target
    match_df_2.drop(columns=["current score"],inplace=True)
    match_df_2.rename(columns={"Current score":"current score"},inplace=True)
    match_df_2["balls"]=(match_df_2["Over"]-1)*6+match_df_2["Ball"]
    match_df_2.drop(columns=["Ball","Innings"],inplace=True)
    match_df=df2.drop(columns=['Ball','Target Score','Date'])
    match_df=match_df[match_df["Match ID"]==match]
    columns=match_df.columns.to_list()
    overs_df=pd.DataFrame(columns=columns)
    initial=match_df.iloc[0].values.tolist()
    initial[5]=0
    initial[7]=120
    initial[8]=10
    overs_df.loc[0]=initial
    cnt=1
    for i in range(5,len(match_df),6):
        overs_df.loc[cnt]=match_df.iloc[i].values.tolist()
        cnt+=1
    over_prog=match_df_2.groupby("Over").sum()[["Runs From Ball","Wicket"]].reset_index()
    runs_per_over=over_prog["Runs From Ball"].to_list()
    overs=overs_df["Over"].to_list()
    wickets=over_prog["Wicket"].to_list()
    wickets=wickets[:-1]
    wickets.insert(0,0)
    overs_df.drop(columns=["Match ID","Chased Successfully","Over"],inplace=True)
    win,lose=[],[]
    columns=overs_df.columns
    for i in range(len(overs_df)):
        record=pd.DataFrame(overs_df.iloc[i])
        values=[[rec[0] for rec in record.values]]
        x=pd.DataFrame(values,columns=columns)
        x.drop(columns=["Unnamed: 0"],inplace=True)
        probabilities=st.session_state.pipe1.predict_proba(x)[0]
        win.append(round(probabilities[1]*100))
        lose.append(round(probabilities[0]*100))
    fig=go.Figure()
    fig.add_trace(go.Bar(x=overs,y=runs_per_over,name="runs per over"))
    fig.add_trace(go.Scatter(x=overs,y=win,mode="markers+lines",marker_color="green",name="win  probability"))
    fig.add_trace(go.Scatter(x=overs,y=lose,mode="markers+lines",marker_color="red",name="loss probability"))
    fig.add_trace(go.Scatter(x=overs,y=wickets,mode="markers+lines",marker_color="yellow",name="wickets per over"))
    fig.update_layout(width=800,height=400,title="Chase progression",xaxis_title="overs",yaxis_title="runs/probability")
    return fig
nav=option_menu(menu_title=None,options=["win percentage calculator","match analysis"],orientation="horizontal")
col1,col2=st.columns([1,1])
if nav=="match analysis":
    with col1: first_innings=st.selectbox("Team batting first",options=df2["Bat First"].unique(),index=None)
    with col2: second_innings=st.selectbox("Team batting second",options=df2["Bat Second"].unique(),index=None)
    valid_date=df2[(df2["Bat First"]==first_innings) & (df2["Bat Second"]==second_innings)]
    with col1: date=st.selectbox("Date",options=valid_date["Date"].unique(),index=None)
    valid_venue=valid_date[valid_date["Date"]==date]
    with col2: venue=st.selectbox("Venue",options=valid_venue["Venue"].unique(),index=None)
    option=st.radio("",options=["innings progression(worm)","match progression (win probability analysis)"],horizontal=True)    
    analyze=st.button("Analyze")
    if analyze and date and first_innings and second_innings and venue:
        match_id=df2[(df2["Date"]==date) & (df2["Bat First"]==first_innings) & (df2["Bat Second"]==second_innings) & (df2["Venue"]==venue)]["Match ID"].unique()[0]
        if option=="innings progression(worm)": fig=innings_progression(match_id,first_innings,second_innings)
        else: fig=match_progression(match_id)
        st.plotly_chart(fig)
else:

    teams=['Pakistan', 'Zimbabwe', 'Bangladesh', 'South Africa', 'Sri Lanka',
        'West Indies', 'India', 'Afghanistan', 'Australia', 'New Zealand',
        'England', 'Ireland', 'Netherlands', 'Nepal']
    venue=['R Premadasa Stadium', 'Sheikh Abu Naser Stadium','Hagley Oval',
        'County Ground', 'Shere Bangla National Stadium', 'Eden Gardens',
        'Barsapara Cricket Stadium', 'New Wanderers Stadium',
        'Sheikh Zayed Stadium', 'Kensington Oval', 'Brian Lara Stadium',
        'Harare Sports Club', 'Beausejour Stadium',
        'Himachal Pradesh Cricket Association Stadium',
        'Civil Service Cricket Club',
        'Saurashtra Cricket Association Stadium',
        'Greater Noida Sports Complex Ground', 'Kennington Oval',
        'Melbourne Cricket Ground','Bellerive Oval',
        'Sharjah Cricket Stadium','Perth Stadium',
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
    st.title("T20 WIN PREDICTOR")
    col1,col2=st.columns(2)
    with col1:
        batting_team=st.selectbox("Select the Batting team",teams)
    with col2:
        bowling_team=st.selectbox("Select the Bowling team",teams)
    selected_stadium=st.selectbox("Select Venue",venue)
    target=st.number_input("Target",step=1,min_value=0)
    col3,col4,col5=st.columns(3)
    with col3:score=st.number_input('Score',step=1)
    with col4:overs=st.number_input('Overs Completed',step=1)
    with col5:wickets=st.number_input('Wickets Out',step=1)
    if st.button("Predict Probability"):
        runs_left=target-score
        balls_left=120-(overs*6)
        wickets=10-wickets
        if overs==0:crr=0
        else:crr=score/overs
        rrr=(runs_left/balls_left)*6
        input_df=pd.DataFrame({'Venue':[selected_stadium], 'Bowling team':[bowling_team], 'Batting team':[batting_team], 'Target Score':[target], 'Runs to Get':[runs_left],
        'Balls Remaining':[balls_left], 'Wickets_Rem':[wickets], 'current runs':[score], 'crr':[crr], 'rrr':[rrr]})
        st.dataframe(input_df)
        result=st.session_state.pipe.predict_proba(input_df)
        loss=result[0][0]
        win=result[0][1]
        st.subheader("Win percentage of " + batting_team + "- " + str(round(win*100)) + "%")
        st.subheader("Win percentage of " + bowling_team + "- " + str(round(loss*100)) + "%")

    
