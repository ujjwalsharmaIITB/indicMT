import streamlit as st
import requests

languages = ['Hindi', 'Bengali', 'Marathi', 'Tamil', 'Kannada', 'English']

st.title("Machine Translation Demo")

# Inline input layout
col1, col2 = st.columns([1, 3])

with col1:
    target_lang = st.selectbox("Select Target Language", languages)

with col2:
    input_sentence = st.text_input("Enter text to translate and click Translate.")

# Initialize translation state
if "translation" not in st.session_state:
    st.session_state.translation = ""

# Dynamic button label
button_label = f"Translate to {target_lang}"

# Translate button and API call
if st.button(button_label) and input_sentence.strip():
    try:
        response = requests.post(
            "https://api.example.com/translate",
            json={"target_lang": target_lang, "text": input_sentence},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        st.session_state.translation = data.get("translation", "No translation found.")
    except requests.exceptions.RequestException as e:
        st.session_state.translation = f"API request failed: {e}"

# Output translation
translation = st.session_state.translation if st.session_state.translation else ""

st.markdown(f"**{translation}**")
# st.code(translation, language=None)
