import React, { useState } from "react";
import axios from "axios";
import "./App.css";

const languages = [
  "English",
  "Hindi",
  "Bengali",
  "Gujarati",
  "Kannada",
  "Marathi",
  "Punjabi",
  "Sanskrit",
  "Tamil",
  "Urdu",
];

const App = () => {
  const [sourceLang, setSourceLang] = useState("English");
  const [targetLang, setTargetLang] = useState("Hindi");
  const [inputText, setInputText] = useState("");
  const [translation, setTranslation] = useState("");

  const handleTranslate = async () => {
    if (!inputText.trim()) return;
    try {
      const res = await axios.post("https://www.cfilt.iitb.ac.in/apisindicMTApp/translate", {
        source_lang: sourceLang,
        target_lang: targetLang,
        text: inputText,
      });
      setTranslation(res.data.translation || "No translation found.");
    } catch (error) {
      setTranslation("API request failed.");
      console.error(error);
    }
  };

  return (
    <div className="app-container">
      <h1 className="title"> IndicMT Translation App </h1>

      <div className="language-selectors">
        <div className="selector">
          <label>Source Language</label>
          <select
            value={sourceLang}
            onChange={(e) => setSourceLang(e.target.value)}
          >
            {languages.map((lang) => (
              <option key={lang} value={lang}>
                {lang}
              </option>
            ))}
          </select>
        </div>
        <div className="selector">
          <label>Target Language</label>
          <select
            value={targetLang}
            onChange={(e) => setTargetLang(e.target.value)}
          >
            {languages.map((lang) => (
              <option key={lang} value={lang}>
                {lang}
              </option>
            ))}
          </select>
        </div>
      </div>

      <textarea
        className="text-area"
        rows={4}
        placeholder="Enter text to translate"
        value={inputText}
        onChange={(e) => setInputText(e.target.value)}
      />

      <div className="button-container">
        <button className="translate-button" onClick={handleTranslate}>
          Translate {sourceLang} → {targetLang}
        </button>
      </div>

      {translation && (
        <div className="translation-display">
          <h3>Translation:</h3>
          <p>{translation}</p>
        </div>
      )}
    </div>
  );
};

export default App;
