import pickle
from pathlib import Path
import pandas as pd 
import plotly.express as px
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit_authenticator as stauth
from streamlit_lottie import st_lottie 
import requests
import json
st.set_page_config(page_title="marketing campaign analysis",layout="wide")
page_by_img = """
<style>
[data-testid="stAppViewContainer"]{
#   background-image : url("https://images.pexels.com/photos/1519088/pexels-photo-1519088.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1"); 
    background-image : url("https://images.pexels.com/photos/1532771/pexels-photo-1532771.jpeg"); 
  background-size: cover ;
}
</style>
"""
st.markdown(page_by_img , unsafe_allow_html=True)
st.sidebar.success("Select a page above. ")
#user authentication
names = ["Sara Ali", "Mariam Ashraf" ]
usernames =["sara1" , "mariam2"]  
file_path= Path(__file__).parent / "hashed_pw.pk1"
with file_path.open("rb") as file:
    hashed_passwords = pickle.load(file)
    
credentials = {
        "usernames":{
            usernames[0]:{
                "name":names[0],
                "password":hashed_passwords[0]
                },
            usernames[1]:{
                "name":names[1],
                "password":hashed_passwords[1]
                }            
            }
        }    

#authenticator = stauth.Authenticate( names, usernames , hashed_passwords, "dashboard", "abcdef" , cookie_expiry_days=30)
authenticator = stauth.Authenticate( credentials, "dashboard", "abcdef" , cookie_expiry_days=30)
name , authentication_status, username = authenticator.login("Login", "main")

if authentication_status == False:
    st.error("username or password is incorrect")
if authentication_status == None:
    st.warning("Please enter username and password")

if authentication_status:


    dataset='bank-additional-full.csv'
    df = pd.read_csv(
        dataset,sep=';'
        ,header=0,
        index_col=False,
        keep_default_na=True
    )

    #El filters
    authenticator.logout("Logout", "sidebar")
    st.sidebar.title(f"Welcome {name}")
    st.sidebar.header("Please Filter Here:")
    age= st.sidebar.multiselect(
        "select the age:",
        options=df["age"].unique(),
        default=df["age"].unique()
    )
    job= st.sidebar.multiselect(
        "select the job:",
        options=df["job"].unique(),
        default=df["job"].unique()
    )
    education= st.sidebar.multiselect(
        "select the education:",
        options=df["education"].unique(),
        default=df["education"].unique()
    )

    df_selection = df.query(
        "age == @age & job == @job & education == @education"
    )
    st.dataframe(df_selection)

    st.title("Conversion for age groups")
    st.markdown("##")

    #KPIS
    df_selection['conversion'] = df_selection['y'].apply(lambda x : 1 if x == 'yes' else 0)
    conversion_rate =( df_selection['conversion'].sum() / df_selection.shape[0] )*100
    conversion_rate_for_ages = ((df_selection.groupby('age')['conversion'].sum()) / (df_selection.groupby('age')['conversion'].count() ))*100.0

    middle_column , right_column= st.columns(2)
    with middle_column:
        st.subheader("conversion rate")
        st.subheader(f"us ${conversion_rate:,}")
    with right_column:
        st.subheader("conversion rate for ages")
        st.subheader(f"{conversion_rate_for_ages}")


    st.markdown("---")

    #bar chart

    #grouping 3ashan el barchart yetla3 mazbot
    df_selection['age_group'] = df_selection['age'].apply(lambda x : '(18 , 30)' if x<30 else '(30 , 40 )' if x<40 else '(40 , 50 )' if x<50 else '(50 , 60 )' if x <60 else '(60 , 70)' if x<70 else '70+' )
    conversion_by_age_group = (df_selection.groupby('age_group')['conversion'].sum() / df_selection.groupby('age_group')['conversion'].count() *100.0
    )
    fig_conversion_by_age = px.bar(
        conversion_by_age_group,
        x=conversion_by_age_group.index,
        y="conversion",
        orientation="v",
        title="<b>Conversion by age</b>",
        color_discrete_sequence=["#0083b8"],
        template="plotly_white"
        )
    st.plotly_chart(fig_conversion_by_age)

    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(df_selection.corr(), annot=True,linewidths=0.5,linecolor="black",fmt=".1f",ax=ax)
    st.write(fig)
    st.set_option('deprecation.showPyplotGlobalUse', False)

    age_marital_df = df_selection.groupby(['age_group' , 'marital'])['conversion'].sum().unstack('marital').fillna(0)

    age_marital_df = age_marital_df.divide( df_selection.groupby('age_group')['conversion'].count(),axis=0 )

    print(age_marital_df.head())    

    ax = age_marital_df.plot(title='age - marital conversion rates' , kind= 'bar' )
    plt.xlabel('age groups')
    plt.ylabel('conversion')
    st.pyplot()
