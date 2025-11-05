# ===============================================================
# 🧠 AI Universal Studio — Versão PRO
# ===============================================================
# Descrição: Sistema multimodal que aprende com texto, imagem e voz
# ===============================================================

import streamlit as st
from transformers import pipeline
from PIL import Image
from deep_translator import GoogleTranslator
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import speech_recognition as sr
from pydub import AudioSegment
import joblib
import os

# ===============================================================
# ⚙️ Configuração da Página
# ===============================================================
st.set_page_config(page_title="AI Universal Studio PRO", page_icon="🧠", layout="wide")

st.title("🧠 AI Universal Studio — Versão PRO")
st.info("""
Sistema de **IA Multimodal** que aprende a partir de **texto**, **imagem** e **voz**  
para gerar previsões inteligentes sobre categorias personalizadas ⚡
""")

# ===============================================================
# 📦 Carregamento do modelo BLIP (image captioning)
# ===============================================================
@st.cache_resource
def load_caption_model():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

captioner = load_caption_model()

# ===============================================================
# 🔁 Sessão Compartilhada
# ===============================================================
for var, default in {
    "keywords": [],
    "categories": [],
    "modelo": None,
    "vectorizer": None
}.items():
    if var not in st.session_state:
        st.session_state[var] = default

# ===============================================================
# 📁 Funções auxiliares
# ===============================================================

def salvar_modelo(modelo, vectorizer):
    joblib.dump(modelo, "modelo_treinado.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")

def carregar_modelo():
    if os.path.exists("modelo_treinado.pkl") and os.path.exists("vectorizer.pkl"):
        modelo = joblib.load("modelo_treinado.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
        return modelo, vectorizer
    return None, None

def transcrever_audio(arquivo):
    try:
        # Converte áudio para formato WAV (SpeechRecognition precisa)
        audio = AudioSegment.from_file(arquivo)
        audio.export("temp.wav", format="wav")
        recognizer = sr.Recognizer()
        with sr.AudioFile("temp.wav") as source:
            audio_data = recognizer.record(source)
            texto = recognizer.recognize_google(audio_data, language="pt-BR")
        os.remove("temp.wav")
        return texto
    except Exception as e:
        return f"[Erro ao processar áudio: {e}]"

# ===============================================================
# 🧭 Abas de Navegação
# ===============================================================
aba = st.tabs([
    "🧩 Etapa 1 - Base de Treinamento",
    "⚙️ Etapa 2 - Treinar Modelo",
    "🔮 Etapa 3 - Fazer Previsão"
])

# ===============================================================
# 1️⃣ ETAPA 1 — Base de Treinamento
# ===============================================================
with aba[0]:
    st.header("🧩 Etapa 1 – Criar base de aprendizado (Palavras + Categorias)")
    st.write("Adicione até **3 exemplos** de texto/frase para ensinar a IA.")

    entradas = []
    for i in range(3):
        col1, col2 = st.columns([3, 1])
        palavras = col1.text_input(f"📝 Exemplo {i+1}:", key=f"texto_{i}")
        categoria = col2.selectbox(
            f"🎯 Categoria {i+1}:",
            ["Baixo", "Moderado", "Alto"],
            index=1,
            key=f"cat_{i}"
        )
        if palavras:
            entradas.append({"texto": palavras, "categoria": categoria})

    if entradas and st.button("💾 Salvar base de aprendizado"):
        st.session_state.keywords = [e["texto"] for e in entradas]
        st.session_state.categories = [e["categoria"] for e in entradas]
        st.success("✅ Base de aprendizado salva com sucesso!")
        st.dataframe(pd.DataFrame(entradas), use_container_width=True)

# ===============================================================
# 2️⃣ ETAPA 2 — Treinar Modelo
# ===============================================================
with aba[1]:
    st.header("⚙️ Etapa 2 – Treinar modelo com base nos exemplos")

    if not st.session_state.keywords or not st.session_state.categories:
        st.warning("⚠️ Nenhum dado de aprendizado. Vá para a Etapa 1 primeiro.")
    else:
        if st.button("🚀 Treinar modelo agora"):
            vectorizer = CountVectorizer(ngram_range=(1, 2))
            X = vectorizer.fit_transform(st.session_state.keywords)
            y = st.session_state.categories
            modelo = RandomForestClassifier(random_state=42)
            modelo.fit(X, y)
            st.session_state.vectorizer = vectorizer
            st.session_state.modelo = modelo
            salvar_modelo(modelo, vectorizer)
            st.success("✅ Modelo treinado e salvo com sucesso! Vá para a Etapa 3.")

        # Se já existir um modelo salvo, carregar automaticamente
        modelo_salvo, vectorizer_salvo = carregar_modelo()
        if modelo_salvo:
            st.session_state.modelo = modelo_salvo
            st.session_state.vectorizer = vectorizer_salvo
            st.info("💾 Modelo salvo carregado automaticamente.")

# ===============================================================
# 3️⃣ ETAPA 3 — Previsão (Imagem + Texto + Áudio)
# ===============================================================
with aba[2]:
    st.header("🔮 Etapa 3 – Fazer previsão com novos dados")
    st.write("Envie uma **imagem**, **áudio** e/ou **texto**, e clique em **Fazer previsão**.")

    col1, col2 = st.columns(2)
    with col1:
        uploaded_img = st.file_uploader("📷 Imagem (opcional):", type=["jpg", "jpeg", "png"])
    with col2:
        uploaded_audio = st.file_uploader("🎤 Áudio (opcional):", type=["mp3", "wav"])

    texto_input = st.text_area("💬 Texto descritivo (opcional):", key="predict_text")

    # --- Processamento da imagem ---
    desc_img = ""
    if uploaded_img:
        image = Image.open(uploaded_img).convert("RGB")
        st.image(image, caption="📸 Imagem enviada", use_container_width=True)
        with st.spinner("🔍 Gerando descrição automática da imagem..."):
            caption_en = captioner(image)[0]["generated_text"]
            desc_img = GoogleTranslator(source="en", target="pt").translate(caption_en)
            st.markdown(f"<small>Descrição da imagem: *{desc_img}*</small>", unsafe_allow_html=True)

    # --- Processamento do áudio real ---
    desc_audio = ""
    if uploaded_audio:
        st.audio(uploaded_audio)
        with st.spinner("🎧 Transcrevendo áudio..."):
            desc_audio = transcrever_audio(uploaded_audio)
            st.markdown(f"<small>Transcrição do áudio: *{desc_audio}*</small>", unsafe_allow_html=True)

    # --- Combina tudo ---
    entrada = f"{desc_img} {desc_audio} {texto_input}".strip()
    st.text_area("🧩 Entrada combinada:", value=entrada, height=120)

    # --- Previsão ---
    if st.button("🔍 Fazer previsão"):
        if not st.session_state.modelo or not st.session_state.vectorizer:
            st.warning("⚠️ Treine o modelo antes de prever.")
        elif not entrada:
            st.warning("⚠️ Insira imagem, áudio ou texto.")
        else:
            X_novo = st.session_state.vectorizer.transform([entrada])
            pred = st.session_state.modelo.predict(X_novo)[0]
            cor = {"Baixo": "green", "Moderado": "orange", "Alto": "red"}[pred]

            exemplos_relacionados = [
                kw for kw, cat in zip(st.session_state.keywords, st.session_state.categories)
                if cat == pred
            ]
            palavra_chave = exemplos_relacionados[0] if exemplos_relacionados else "N/A"

            st.markdown(
                f"""
                <div style='background-color:#f0f2f6;padding:20px;border-radius:12px;text-align:center;'>
                    <h3>🧠 Previsão da IA: <span style='color:{cor};'>{pred}</span></h3>
                    <p style='font-size:18px;color:gray;'>🔑 Palavra-chave associada: <b>{palavra_chave}</b></p>
                </div>
                """,
                unsafe_allow_html=True
            )
