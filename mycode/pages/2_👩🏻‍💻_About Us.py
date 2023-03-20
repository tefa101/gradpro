import json
import requests
import streamlit as st 
from streamlit_lottie import st_lottie 

st.subheader("About Us")

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
    
def load_lottieurl(url : str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_coding = load_lottiefile("lottiefile/coding4.json")
st_lottie(
    lottie_coding,
    height=350,
    width=500
  
)
st.title("Our Mission")

st.write("""
              #### Marketing research is usually the first step in the marketing process, after ideas for products are conceived. Businesses conduct marketing research to obtain information from the marketplace. They use it to solve problems, obtain information on competitors and determine the needs and wants of non-paying consumers and customers. Marketers then analyze the data and develop various marketing strategies.so our aim is to make powerful analysis accessible to anyone. Interactive reports let you query your data with only a few clicks, then see visualizations. This makes it easy to answer question after question about how your product is used, who sticks around, and much more.     
            
            """)


st.markdown("---")


st.title("What We Offer?")
st.write("""
         
    ### 📊 Interactive Dashboards

    ### 👥 Customer Segmentation

    ### 🔐 Security For Your Data

    ### ⏳ Fast Time Processing
         
         """)

st.markdown("---")