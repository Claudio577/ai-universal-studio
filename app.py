import streamlit as st
from transformers import pipeline
from PIL import Image
from deep_translator import GoogleTranslator
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
import random

# ==============================
# ⚙️ Configuração inicial
# ==============================
st.set_page_config(page_title="AI Universal Studio", page_icon="🧠", layout="wide")
st.title("🧠 AI Universal Studio")
st.write("Sistema de IA genérico que analisa **imagens**, **textos** e **planilhas (CSV)** para gerar **previsões e análises inteligentes** ⚡")

# ==============================
# 🧩 Modelos
# ==============================
@st.cache_resource
def load_caption_model():
    try:
        model_name = "microsoft/git-large-coco"
        captioner = pipeline("image-to-text", model=model_name)
        return captioner
    except:
        return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

@st.cache_resource
def load_text_model():
    try:
        return pipeline("text2text-generation", model="google/flan-t5-small")
    except:
        return None

captioner = load_caption_model()
refiner = load_text_model()

# ==============================
# 🧭 Abas principais
# ==============================
aba = st.tabs([
    "🧩 Treinar IA",
    "🧠 Previsão (Imagem + Texto)",
    "📋 Explicação"
])

# ==============================
# 🔁 Sessão compartilhada
# ==============================
if "modelo" not in st.session_state:
    st.session_state.modelo = None
if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None
if "img_desc" not in st.session_state:
    st.session_state.img_desc = ""
if "txt_desc" not in st.session_state:
    st.session_state.txt_desc = ""

# ======================================================
# 🧩 1 – Treinar IA (Palavras ou CSV)
# ======================================================
with aba[0]:
    st.header("🧩 Treinamento Rápido da IA")
    st.write("Você pode treinar a IA com **palavras-chave** ou **planilha CSV**.")

    modo = st.radio("Escolha o tipo de treinamento:", ["Palavras-Chave", "Arquivo CSV"])

    if modo == "Palavras-Chave":
        n = st.number_input("Quantos grupos de palavras deseja adicionar?", 1, 10, 3)
        entradas = []
        for i in range(n):
            col1, col2 = st.columns([3, 1])
            palavras = col1.text_input(f"Palavras/frases do grupo {i+1}:")
            categoria = col2.text_input(f"Categoria {i+1}:")
            if palavras and categoria:
                entradas.append({"texto": palavras, "categoria": categoria})

        if len(entradas) > 1:
            df = pd.DataFrame(entradas)
            st.dataframe(df)
            if st.button("🚀 Treinar modelo"):
                vectorizer = CountVectorizer(stop_words="portuguese")
                X = vectorizer.fit_transform(df["texto"])
                y = df["categoria"]

                modelo = RandomForestClassifier()
                modelo.fit(X, y)
                st.session_state.vectorizer = vectorizer
                st.session_state.modelo = modelo
                st.success("✅ Modelo treinado com sucesso!")

    else:
        uploaded_csv = st.file_uploader("📎 Envie seu arquivo CSV", type=["csv"])
        if uploaded_csv:
            df = pd.read_csv(uploaded_csv)
            st.dataframe(df.head())
            target = st.selectbox("🎯 Escolha a coluna de resultado:", df.columns)
            if st.button("🚀 Treinar modelo com CSV"):
                X = df.drop(columns=[target])
                y = df[target]
                X = pd.get_dummies(X)
                modelo = RandomForestClassifier()
                modelo.fit(X, y)
                st.session_state.modelo = modelo
                st.success("✅ Modelo treinado com sucesso!")

# ======================================================
# 🧠 2 – Previsão Multimodal
# ======================================================
with aba[1]:
    st.header("🧠 Previsão com Imagem + Texto")

    uploaded_img = st.file_uploader("📷 Envie uma imagem (opcional)", type=["jpg", "jpeg", "png"])
    texto_input = st.text_area("💬 Escreva ou cole um texto (opcional):")

    if uploaded_img or texto_input:
        desc_img = ""
        if uploaded_img:
            image = Image.open(uploaded_img).convert("RGB")
            st.image(image, caption="📸 Imagem enviada", use_container_width=True)
            caption_en = captioner(image)[0]["generated_text"]
            desc_img = GoogleTranslator(source="en", target="pt").translate(caption_en)

        entrada_unificada = f"{desc_img} {texto_input}".strip()
        st.text_area("🧩 Entrada combinada:", value=entrada_unificada, height=120)

        if st.session_state.modelo and st.session_state.vectorizer:
            X_novo = st.session_state.vectorizer.transform([entrada_unificada])
            pred = st.session_state.modelo.predict(X_novo)[0]
            st.success(f"🧠 Previsão automática: **{pred}**")
        else:
            st.warning("⚠️ Treine um modelo primeiro na aba anterior.")

# ======================================================
# 📋 3 – Explicação
# ======================================================
with aba[2]:
    st.header("📋 Explicação e Justificativa da Previsão")
    if st.session_state.modelo and refiner:
        img_desc = st.text_area("📷 Descrição automática da imagem:", value=st.session_state.img_desc, height=100)
        txt_desc = st.text_area("🩺 Texto clínico ou observações:", value=st.session_state.txt_desc, height=100)

        if st.button("🔍 Gerar explicação"):
            combinado = (img_desc.strip() + " " + txt_desc.strip()).strip()
            prompt = f"Explique o seguinte caso de forma profissional e ética, classificando o risco:\n\n{combinado}"
            explicacao = refiner(prompt, max_new_tokens=120)[0]["generated_text"]
            st.write(explicacao)
    else:
        st.info("⚠️ Treine um modelo e gere uma previsão antes de pedir explicação.")

