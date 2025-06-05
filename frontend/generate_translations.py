
import sentencepiece as spm
import argparse
import ctranslate2



spm_model_file = "models/vocab/multilingual.model"
spm_model = spm.SentencePieceProcessor()
spm_model.Load(spm_model_file)



ctranslate2_model_file = "models/multilingual_160k"

translator = ctranslate2.Translator(ctranslate2_model_file, inter_threads=4, intra_threads=1)


languages_to_tokens = {
    "english": "<en>",
    "hindi": "<hi>",
    "marathi": "<mr>",
    "bengali": "<bn>",
    "gujarati": "<gu>",
    "kannada": "<kn>",
    "punjabi": "<pa>",
    "sanskrit": "<sa>",
    "tamil": "<ta>",
    "urdu": "<ur>",
}

def tokenize(sentence):
    return spm_model.EncodeAsPieces(sentence.strip())

def detokenize(spm_sentence):
    return spm_model.DecodePieces(spm_sentence)



def add_token(sentence, language):
    token = languages_to_tokens.get(language.lower())
    return token + " " + sentence



def translate_en_xx_en(sentence, language):
    tgt_token_added_sentence = add_token(sentence, language)
    spm_tokenized_sentence = tokenize(tgt_token_added_sentence)
    spm_tokenized_sentence = [spm_tokenized_sentence]
    out_lines = translator.translate_batch(spm_tokenized_sentence, beam_size=5, max_batch_size=16)
    out_line = out_lines[0].hypotheses[0]
    out_line = detokenize(out_line)
    out_line = out_line.replace('"', '').replace("u200d", "").strip()
    return out_line


def translate_indic_indic(sentence, language):
    # pivot based translation
    # add English token to the sentence
    english_translated_sentence = translate_en_xx_en(sentence, "english")
    # Translate english sentence to indic language
    tgt_language_translated_sentence = translate_en_xx_en(english_translated_sentence, language)
    return tgt_language_translated_sentence



def translate(sentence, source_language, target_language):
    if source_language.lower() == "english":
        return translate_en_xx_en(sentence, target_language)
    elif target_language.lower() == "english":
        return translate_en_xx_en(sentence, source_language)
    else:
        return translate_indic_indic(sentence, target_language)

# def translate(sentence, language):
#     tgt_token_added_sentence = add_token(sentence, language)
#     spm_tokenized_sentence = tokenize(tgt_token_added_sentence)
#     spm_tokenized_sentence = [spm_tokenized_sentence]
#     out_lines = translator.translate_batch(spm_tokenized_sentence, beam_size=5, max_batch_size=16)
#     out_line = out_lines[0].hypotheses[0]
#     out_line = detokenize(out_line)
#     out_line = out_line.replace('"', '').replace("u200d", "").strip()
#     return out_line





# print(translate_en_xx_en("Hello, how are you?", "Hindi"))
# print(translate_en_xx_en("Hello, how are you?", "Bengali"))
# print(translate_en_xx_en("Hello, how are you?", "Tamil"))
# print(translate_en_xx_en("Hello, how are you?", "Marathi"))




# print(translate_indic_indic("नमस्कार, आप कैसे हैं?", "Tamil"))
