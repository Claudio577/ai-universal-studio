import streamlit as st
from transformers import pipeline
from PIL import Image
from deep_translator import GoogleTranslator
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

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
# 🔁 Sessão compartilhada
# ==============================
for var, default in {
    "keywords": [],
    "modelo": None,
    "vectorizer": None,
    "img_desc": "",
    "txt_desc": ""
}.items():
    if var not in st.session_state:
        st.session_state[var] = default

# ==============================
# 🧭 Abas
# ==============================
aba = st.tabs([
    "🧩 Palavras-Chave / CSV",
    "🧠 Previsão (Imagem + Texto)",
    "📋 Explicação"
])

# ======================================================
# 1️⃣ PALAVRAS-CHAVE / CSV  → apenas gera base
# ======================================================
with aba[0]:
    st.header("🧩 Geração de Palavras-Chave")
    modo = st.radio("Escolha o tipo de entrada:", ["Palavras-Chave", "Arquivo CSV"])

    if modo == "Palavras-Chave":
        n = st.number_input("Quantos grupos deseja adicionar?", 1, 10, 3)
        entradas = []
        for i in range(n):
            col1, col2 = st.columns([3, 1])
            palavras = col1.text_input(f"Palavras/frases do grupo {i+1}:")
            categoria = col2.text_input(f"Categoria {i+1}:")
            if palavras:
                entradas.append(palavras)
        if entradas and st.button("💾 Salvar palavras-chave"):
            st.session_state.keywords = entradas
            st.success("✅ Palavras-chave salvas para uso no treinamento!")
            st.write(entradas)

    else:
        uploaded_csv = st.file_uploader("📎 Envie seu CSV (para extrair palavras-chave)", type=["csv"])
        if uploaded_csv:
            df = pd.read_csv(uploaded_csv)
            st.dataframe(df.head())
            col_escolhida = st.selectbox("Escolha a coluna de texto:", df.columns)
            if st.button("💾 Extrair palavras-chave"):
                palavras = df[col_escolhida].dropna().astype(str).tolist()
                st.session_state.keywords = palavras
                st.success("✅ Palavras-chave extraídas!")
                st.write(palavras[:10])

# ======================================================
# 2️⃣ PREVISÃO / TREINAMENTO REAL
# ======================================================
with aba[1]:
    st.header("🧠 Treinar e Prever com Imagem + Texto")

    uploaded_img = st.file_uploader("📷 Envie uma imagem (opcional)", type=["jpg", "jpeg", "png"])
    texto_input = st.text_area("💬 Texto descritivo (opcional):")

    if uploaded_img or texto_input:
        desc_img = ""
        if uploaded_img:
            image = Image.open(uploaded_img).convert("RGB")
            st.image(image, caption="📸 Imagem enviada", use_container_width=True)
            caption_en = captioner(image)[0]["generated_text"]
            desc_img = GoogleTranslator(source="en", target="pt").translate(caption_en)

        entrada = f"{desc_img} {texto_input}".strip()
        st.text_area("🧩 Entrada combinada:", value=entrada, height=120)

        # --- Treinamento se ainda não houver modelo ---
        if st.button("🚀 Treinar modelo com base atual"):
            if not st.session_state.keywords:
                st.warning("⚠️ Nenhuma palavra-chave carregada. Vá à aba anterior primeiro.")
            else:
                # usa as palavras-chave como dados de treino
                textos = st.session_state.keywords
                categorias = ["base"] * len(textos)
                vectorizer = CountVectorizer()
                X = vectorizer.fit_transform(textos)
                modelo = RandomForestClassifier()
                modelo.fit(X, categorias)
                st.session_state.vectorizer = vectorizer
                st.session_state.modelo = modelo
                st.success("✅ Modelo treinado com base nas palavras-chave!")

        # --- Previsão se modelo existir ---
        if st.session_state.modelo and st.session_state.vectorizer:
            X_novo = st.session_state.vectorizer.transform([entrada])
            pred = st.session_state.modelo.predict(X_novo)[0]
            st.success(f"🧠 Previsão automática: **{pred}**")
        else:
            st.info("ℹ️ Treine o modelo primeiro para habilitar a previsão.")

# ======================================================
# 3️⃣ EXPLICAÇÃO / JUSTIFICATIVA
# ======================================================
with aba[2]:
    st.header("📋 Explicação e Justificativa da Previsão")
    img_desc = st.text_area("📷 Descrição automática da imagem:", value=st.session_state.img_desc, height=100)
    txt_desc = st.text_area("🩺 Texto clínico ou observações:", value=st.session_state.txt_desc, height=100)

    if st.button("🔍 Gerar explicação"):
        combinado = (img_desc.strip() + " " + txt_desc.strip()).strip()
        if not refiner:
            st.warning("⚠️ Modelo de explicação não carregado.")
        elif not combinado:
            st.warning("Insira pelo menos um texto para gerar explicação.")
        else:
            with st.spinner("🧠 Gerando explicação com IA..."):
                prompt = f"Explique o seguinte caso e classifique o risco:\n\n{combinado}"
                explicacao = refiner(prompt, max_new_tokens=120)[0]["generated_text"]
                st.success("✅ Explicação gerada:")
                st.write(explicacao)

