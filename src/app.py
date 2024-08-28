import streamlit as st
import sys
from model import predict
from LLM import chain
from PIL import Image

sys.path.append("...")

st.set_page_config(page_title='🦜🔗 Quick GPT')
st.title("Quick GPT 📖📚🏫📝")

img = st.file_uploader("Please insert an image: ", type=['png', 'jpeg', 'jpg'])

query_input = st.text_area(f"What do you want to know about?", disabled=not img)
submitted = st.button('Submit')

if img:
    breed = Image.open(img)
    st.write(f"Breed Prediciton: {predict(breed)}")
# Process the query with a spinner
with st.spinner("Thinking about it..."):
    if submitted:
        st.write(f'Answer: {chain.invoke({"question": query_input, "breed": breed})}')
