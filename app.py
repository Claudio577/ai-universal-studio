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
st.write("Demonstração de um sistema de IA que aprende a partir de **imagens** e **textos** para gerar **previsões inteligentes** ⚡")

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
    "vectorizer": None
}.items():
    if var not in st.session_state:
        st.session_state[var] = default

# ==============================
# 🧭 Abas
# ==============================
aba = st.tabs([
    "🧩 Etapa 1 - Base de Treinamento",
    "🧠 Etapa 2 - Treinar e Prever"
])

# ======================================================
# 1️⃣ ETAPA 1 – BASE DE TREINAMENTO
# ======================================================
with aba[0]:
    st.header("🧩 Etapa 1 – Criar base de aprendizado (Palavras + Categorias)")
    st.write("Adicione até **3 exemplos de texto** para ensinar a IA o que significa cada categoria (Baixo, Moderado, Alto risco).")

    entradas = []
    for i in range(3):
        col1, col2 = st.columns([3, 1])
        palavras = col1.text_input(f"📝 Exemplo {i+1} (texto ou frase):", key=f"texto_{i}")
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
        st.dataframe(pd.DataFrame(entradas))

# ======================================================
# 2️⃣ ETAPA 2 – TREINAR E PREVER (Imagem + Texto)
# ======================================================
with aba[1]:
    st.header("🧠 Etapa 2 – Treinar modelo e realizar previsões")
    st.write("Envie uma **imagem** e/ou **texto descritivo**, e a IA fará a previsão com base nos exemplos anteriores.")

    uploaded_img = st.file_uploader("📷 Envie uma imagem (opcional):", type=["jpg", "jpeg", "png"])
    texto_input = st.text_area("💬 Texto descritivo (opcional):")

    if uploaded_img or texto_input:
        desc_img = ""
        if uploaded_img:
            image = Image.open(uploaded_img).convert("RGB")
            st.image(image, caption="📸 Imagem enviada", use_container_width=True)
            with st.spinner("🔍 Gerando descrição automática da imagem..."):
                caption_en = captioner(image)[0]["generated_text"]
                desc_img = GoogleTranslator(source="en", target="pt").translate(caption_en)

        entrada = f"{desc_img} {texto_input}".strip()
        st.text_area("🧩 Entrada combinada para previsão:", value=entrada, height=120)

        # --- Treinamento rápido ---
        if st.button("🚀 Treinar modelo"):
            if not st.session_state.keywords or not st.session_state.categories:
                st.warning("⚠️ Nenhum dado de aprendizado. Vá para a Etapa 1 primeiro.")
            else:
                vectorizer = CountVectorizer()
                X = vectorizer.fit_transform(st.session_state.keywords)
                y = st.session_state.categories
                modelo = RandomForestClassifier()
                modelo.fit(X, y)
                st.session_state.vectorizer = vectorizer
                st.session_state.modelo = modelo
                st.success("✅ Modelo treinado com sucesso!")

        # --- Previsão ---
        if st.session_state.modelo and st.session_state.vectorizer and entrada:
            X_novo = st.session_state.vectorizer.transform([entrada])
            pred = st.session_state.modelo.predict(X_novo)[0]
            cor = {"Baixo": "green", "Moderado": "orange", "Alto": "red"}[pred]

            st.markdown(
                f"<h3>🧠 Previsão da IA: <span style='color:{cor}'>{pred}</span></h3>",
                unsafe_allow_html=True
            )

            exemplos_relacionados = [
                kw for kw, cat in zip(st.session_state.keywords, st.session_state.categories)
                if cat == pred
            ]
            if exemplos_relacionados:
                st.markdown("📚 **Exemplos usados para essa categoria:**")
                st.write(exemplos_relacionados)
        else:
            st.info("ℹ️ Treine o modelo primeiro antes de prever.")
