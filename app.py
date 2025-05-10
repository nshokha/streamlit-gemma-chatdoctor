import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from huggingface_hub import login

token = 'your_token'
model_dir = './models/ChatDoctor'

def load_med_model():
    if not os.path.exists(model_dir):
        with st.spinner('Downloading model...'):
            login(token)
            model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it")
            model.save_pretrained(model_dir)
            st.success('Model is downloaded!')
    else:
        with st.spinner('Loading model...'):
            model = AutoModelForCausalLM.from_pretrained(model_dir)
            st.success('Model is loaded!')
    return model


if 'chatdoctor' not in st.session_state:
    st.session_state['chatdoctor'] = None

st.set_page_config('Multi-functional AI', layout="wide", initial_sidebar_state="expanded")

with st.sidebar:
    st.markdown("Your personal AI assistant")
    menu = ['Chat Doctor', 'TTS', 'STT', 'Voice Cloning', 'RAG', 'ChatGPT']
    choices = st.selectbox('Models:', menu)

if choices == "Chat Doctor":
    st.title("Medical AI ChatDoctor")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.header("Medical Model Info")
        st.write("This section uses the 'Gemma-2-2b-it-ChatDoctor' model.")

    with col2:
        st.header("Load Medical Model")
        if st.button("Load Model"):
            st.session_state['chatdoctor'] = load_med_model()

    with col3:
        st.header("Model Interaction")
        if st.session_state['chatdoctor'] is None:
            st.info("Please load the medical model first.")
        else:
            user_input = st.text_input("Ask a medical question:")
            if user_input:
                with st.spinner("Generating response..."):
                    try:
                        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
                        inputs = tokenizer(user_input, return_tensors="pt")
                        response_ids = st.session_state['medical_model'].generate(
                            **inputs, max_length=100, num_return_sequences=1, temperature=0.7
                        )
                        response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
                        st.success("Response:")
                        st.write(response)
                    except Exception as e:
                        st.error(f"An error occurred while generating the response: {e}")
