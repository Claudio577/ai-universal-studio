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

captioner = load_caption_model()

# ==============================
# 🔁 Sessão compartilhada
# ==============================
for var, default in {
    "keywords": [],
    "categories": [],
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
    "🧠 Previsão (Imagem + Texto)"
])

# ======================================================
# 1️⃣ PALAVRAS-CHAVE / CSV → base de aprendizado
# ======================================================
with aba[0]:
    st.header("🧩 Geração de Palavras-Chave e Categorias")
    modo = st.radio("Escolha o tipo de entrada:", ["Palavras-Chave", "Arquivo CSV"])

    if modo == "Palavras-Chave":
        n = st.number_input("Quantos grupos deseja adicionar?", 1, 10, 3)
        entradas = []
        for i in range(n):
            col1, col2 = st.columns([3, 1])
            palavras = col1.text_input(f"Palavras/frases do grupo {i+1}:")
            categoria = col2.selectbox(
                f"Categoria {i+1}:",
                ["Baixo", "Moderado", "Alto"],
                index=1,
                key=f"cat_{i}"
            )
            if palavras:
                entradas.append({"texto": palavras, "categoria": categoria})
        if entradas and st.button("💾 Salvar palavras-chave"):
            st.session_state.keywords = [e["texto"] for e in entradas]
            st.session_state.categories = [e["categoria"] for e in entradas]
            st.success("✅ Palavras-chave e categorias salvas com sucesso!")
            st.dataframe(pd.DataFrame(entradas))

    else:
        uploaded_csv = st.file_uploader("📎 Envie seu CSV (para extrair palavras-chave)", type=["csv"])
        if uploaded_csv:
            df = pd.read_csv(uploaded_csv)
            st.dataframe(df.head())
            col_text = st.selectbox("Coluna de texto:", df.columns)
            col_cat = st.selectbox("Coluna de categoria:", df.columns)
            if st.button("💾 Extrair dados"):
                st.session_state.keywords = df[col_text].dropna().astype(str).tolist()
                st.session_state.categories = df[col_cat].dropna().astype(str).tolist()
                st.success("✅ Palavras-chave e categorias extraídas!")
                st.write(st.session_state.keywords[:5])

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

        # --- Treinamento ---
        if st.button("🚀 Treinar modelo com base atual"):
            if not st.session_state.keywords or not st.session_state.categories:
                st.warning("⚠️ Nenhum dado de treinamento. Vá à aba anterior.")
            else:
                vectorizer = CountVectorizer()
                X = vectorizer.fit_transform(st.session_state.keywords)
                y = st.session_state.categories
                modelo = RandomForestClassifier()
                modelo.fit(X, y)
                st.session_state.vectorizer = vectorizer
                st.session_state.modelo = modelo
                st.success("✅ Modelo treinado com sucesso com base nas palavras e categorias!")

        # --- Previsão ---
        if st.session_state.modelo and st.session_state.vectorizer:
            X_novo = st.session_state.vectorizer.transform([entrada])
            pred = st.session_state.modelo.predict(X_novo)[0]

            # Define cor de acordo com o risco
            cor = {"Baixo": "green", "Moderado": "orange", "Alto": "red"}[pred]

            st.markdown(
                f"<h4>🧠 Previsão automática: <span style='color:{cor}'>{pred}</span></h4>",
                unsafe_allow_html=True
            )

            # Exemplo que influenciou
            exemplos_relacionados = [
                kw for kw, cat in zip(st.session_state.keywords, st.session_state.categories)
                if cat == pred
            ]
            st.write(f"📚 Exemplos que ajudaram nessa previsão ({pred}):")
            st.json(exemplos_relacionados[:3])
        else:
            st.info("ℹ️ Treine o modelo primeiro para habilitar a previsão.")
