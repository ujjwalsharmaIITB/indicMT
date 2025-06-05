import streamlit as st
import requests


source_languages = [
    'English',
    'Hindi', 
    'Bengali', 
    'Gujarati', 
    'Kannada', 
    'Marathi', 
    'Punjabi', 
    'Sanskrit', 
    'Tamil', 
    'Urdu']

target_languages = [
    'Hindi',
    'English', 
    'Bengali', 
    'Gujarati', 
    'Kannada', 
    'Marathi', 
    'Punjabi', 
    'Sanskrit', 
    'Tamil', 
    'Urdu']
st.title("Machine Translation Demo")


@st.cache_resource
def load_translate():
    from generate_translations import translate
    return translate
    


translate = load_translate()

translated_sentence = None

# Inline input layout
col1, col2, col3 = st.columns([1, 1, 3])

with col1:
    source_lang = st.selectbox("Source Language", source_languages, index=0)

with col2:
    target_lang = st.selectbox("Target Language", target_languages, index=0)

with col3:
    input_sentence = st.text_input("Enter text to translate and click Translate.")

# Initialize translation state
if "translation" not in st.session_state:
    st.session_state.translation = ""

# Dynamic button label
button_label = f"Translate {source_lang} → {target_lang}"

# Translate button and API call
if st.button(button_label) and input_sentence.strip():

    translated_sentence = translate(input_sentence, source_lang, target_lang)
    
    # try:
    #     response = requests.post(
    #         "https://api.example.com/translate",
    #         json={
    #             "source_lang": source_lang,
    #             "target_lang": target_lang,
    #             "text": input_sentence
    #         },
    #         timeout=10,
    #     )
    #     response.raise_for_status()
    #     data = response.json()
    #     st.session_state.translation = data.get("translation", "No translation found.")
    # except requests.exceptions.RequestException as e:
    #     st.session_state.translation = f"API request failed: {e}"


# Output translation
translation = translated_sentence if translated_sentence else ""

st.markdown(f"**{translation}**")
