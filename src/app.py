import streamlit as st
import sys
from model import predict
from LLM import chain
from eval import rag_chain
from PIL import Image
import base64

st.set_page_config(page_title='🐕🔗 Dog LLM')
st.title("Dog LLM 🐕")
st.sidebar.title("")

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    [data-testid=stSidebar] {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    background-position: center;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background(r"C:\Users\edgar\OneDrive\Ambiente de Trabalho\AI_Projects\Dog_LLM\image\Becca-Jackabee.jpg")

img = st.file_uploader("Please insert an image: ", type=['png', 'jpeg', 'jpg'])

query_input = st.text_area(f"What do you want to know about?", disabled=not img)
col1, col2 = st.columns([0.05, 0.5])
submitted = col1.button('RAG')
filter =col2.button('Filtered RAG')

if img:
    breed = Image.open(img)
    st.write(f"Breed Prediciton: {predict(breed)}")
# Process the query with a spinner
with st.spinner("Thinking about it..."):
    if submitted:
        st.write(f'Answer: {chain.invoke({"question": query_input, "breed": breed})}')
    elif filter:
        st.write(f'Answer: {rag_chain.invoke({"question": query_input, "breed": breed})}')