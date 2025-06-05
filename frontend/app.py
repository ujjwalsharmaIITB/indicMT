import streamlit as st
import requests

# Define language lists
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
    'Urdu'
]

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
    'Urdu'
]

st.set_page_config(page_title="Indic MT")

# Centered title with Indian flag
st.markdown(
    "<h1 style='text-align: center;'> IndicMT Translation App </h1>",
    unsafe_allow_html=True
)

# Load the translation function (cached)
@st.cache_resource
def load_translate():
    from generate_translations import translate
    return translate

translate = load_translate()
translated_sentence = None

# Language selectors
col1, col2 = st.columns([1, 1])
with col1:
    source_lang = st.selectbox("Source Language", source_languages, index=0)
with col2:
    target_lang = st.selectbox("Target Language", target_languages, index=0)

# Input area
input_sentence = st.text_area("Enter text to translate:", height=68)

# Dynamic button label
button_label = f"Translate {source_lang} → {target_lang}"

# Centered button
col_btn = st.columns([2, 1, 2])
with col_btn[1]:
    if st.button(button_label) and input_sentence.strip():
        translated_sentence = translate(input_sentence, source_lang, target_lang)

# Show translation
translation = translated_sentence if translated_sentence else ""
# Show translation in larger font using styled HTML
if translated_sentence:
    st.write(f"<div style='text-align: center; font-size: 24px; font-weight: bold; color: #FFF;'>{translated_sentence}</div>", unsafe_allow_html=True)




# Added example for API Call

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
