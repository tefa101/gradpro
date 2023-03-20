import streamlit as st 
import json
import requests
from streamlit_lottie import st_lottie 

st.subheader("Contact Us")

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
    
def load_lottieurl(url : str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_coding = load_lottiefile("lottiefile/coding5.json")
st_lottie(
    lottie_coding,
    height=350,
    width=500
)

st.title("let's meet together")

st.write("""##### Please feel free to contact us when facing any troubles in order to offer you the best experience , you can contact via email or phone number or visit us at our location.
         """)


st.write("""### 📍 Location""")
st.text(" Street Name 50,Building 85,12345 SoHo,EG")
st.write("""### ⏰ Opening Hours""")
st.text(" Monday to Friday 10am to 7pm")
st.write("""### 📩 E-mail""")
st.text(" Mail@youcompany.com")
st.write("""### 📞 Phone""")
st.text("01*********")
st.markdown("---")

st.title(":mailbox: Get In Touch With Me!")
contact_form="""
<form action = "https://formsubmit.co/your@email.com" method="POST">
<input type= "hidden" name="_captcha" value="false">
     <input type="text" name="name" placeholder="Your name" required>
     <input type="email" name="email" placeholder="Your email" required>
     <textarea name="message" placeholder="Your message here"></textarea>
     <button type = "submit" >send</button>
</form>
"""
st.markdown(contact_form , unsafe_allow_html=True)
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>" , unsafe_allow_html=True)
local_css("style/style.css")

st.markdown("---")

st.title("Our Social Media Links")




