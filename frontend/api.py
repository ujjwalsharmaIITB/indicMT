from flask import Flask, request, jsonify
from generate_translations import translate
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

print("Running")
@app.route("/translate", methods=["POST"])
def translate_api():
    data = request.get_json()
    sentence = data.get("text", "")
    source_lang = data.get("source_lang", "")
    target_lang = data.get("target_lang", "")
    
    if not sentence or not source_lang or not target_lang:
        return jsonify({"error": "Invalid input"}), 400
    
    try:
        translation = translate(sentence, source_lang, target_lang)
        return jsonify({"translation": translation})
    except Exception as e:
        return jsonify({"error": str(e)}), 500



# run.py

from waitress import serve


if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=8892, url_prefix="/apisindicMTApp")