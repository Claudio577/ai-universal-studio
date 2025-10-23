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
st.write("Sistema de IA genérico que analisa **imagens**, **textos** e **palavras-chave (CSV)** para gerar **previsões e explicações inteligentes** ⚡")

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
    "keywords": pd.DataFrame(),
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
# 1️⃣ PALAVRAS-CHAVE / CSV → GERA BASE
# ======================================================
with aba[0]:
    st.header("🧩 Geração de Palavras-Chave e Categorias")
    st.write("""
    Adicione palavras e **associe uma categoria** a cada grupo.  
    Exemplos de categorias possíveis:  
    🟢 **Baixo** — animal saudável ou comportamento normal  
    🟡 **Moderado** — sintomas leves ou comportamento alterado  
    🔴 **Alto** — caso crítico, sinais graves ou de risco  
    """)

    modo = st.radio("Escolha o tipo de entrada:", ["Palavras-Chave", "Arquivo CSV"])

    if modo == "Palavras-Chave":
        n = st.number_input("Quantos grupos deseja adicionar?", 1, 10, 3)
        entradas = []
        for i in range(n):
            col1, col2 = st.columns([3, 1])
            palavras = col1.text_input(f"Palavras/frases do grupo {i+1}:")
            categoria = col2.selectbox(
                f"Categoria {i+1}:",
                ["", "Baixo", "Moderado", "Alto"],
                key=f"cat_{i}"
            )
            if palavras and categoria:
                entradas.append({"texto": palavras, "categoria": categoria})

        if entradas and st.button("💾 Salvar palavras-chave"):
            df = pd.DataFrame(entradas)
            st.session_state.keywords = df
            st.success("✅ Palavras-chave e categorias salvas com sucesso!")
            st.dataframe(df)

    else:
        uploaded_csv = st.file_uploader("📎 Envie seu CSV (para extrair palavras-chave)", type=["csv"])
        if uploaded_csv:
            df = pd.read_csv(uploaded_csv)
            st.dataframe(df.head())
            col_escolhida = st.selectbox("Escolha a coluna de texto:", df.columns)
            if st.button("💾 Extrair palavras-chave"):
                palavras = df[col_escolhida].dropna().astype(str).tolist()
                st.session_state.keywords = pd.DataFrame({"texto": palavras, "categoria": ["Base"] * len(palavras)})
                st.success("✅ Palavras-chave extraídas e armazenadas!")
                st.write(st.session_state.keywords.head())

# ======================================================
# 2️⃣ PREVISÃO / TREINAMENTO REAL
# ======================================================
with aba[1]:
    st.header("🧠 Treinar e Prever com Imagem + Texto")

    uploaded_img = st.file_uploader("📷 Envie uma imagem (opcional)", type=["jpg", "jpeg", "png"])
    texto_input = st.text_area("💬 Texto descritivo (opcional):")

    entrada_final = ""
    if uploaded_img:
        image = Image.open(uploaded_img).convert("RGB")
        st.image(image, caption="📸 Imagem enviada", use_container_width=True)
        caption_en = captioner(image)[0]["generated_text"]
        desc_img = GoogleTranslator(source="en", target="pt").translate(caption_en)
        entrada_final += " " + desc_img
    if texto_input:
        entrada_final += " " + texto_input

    entrada_final = entrada_final.strip()
    st.text_area("🧩 Entrada combinada:", value=entrada_final, height=120)

    # --- Treinamento ---
    if st.button("🚀 Treinar modelo com base atual"):
        if st.session_state.keywords is None or st.session_state.keywords.empty:
            st.warning("⚠️ Nenhuma palavra-chave carregada. Vá à aba anterior primeiro.")
        else:
            df = st.session_state.keywords.copy()
            df["texto"] = df["texto"].astype(str).fillna("")
            df["categoria"] = df["categoria"].astype(str).fillna("")

            textos = df["texto"].tolist()
            categorias = df["categoria"].tolist()

            vectorizer = CountVectorizer()
            X = vectorizer.fit_transform(textos)
            modelo = RandomForestClassifier()
            modelo.fit(X, categorias)

            st.session_state.vectorizer = vectorizer
            st.session_state.modelo = modelo
            st.success("✅ Modelo treinado com sucesso com base nas palavras e categorias!")

    # --- Previsão ---
    if st.session_state.modelo and st.session_state.vectorizer:
        if entrada_final:
            X_novo = st.session_state.vectorizer.transform([entrada_final])
            pred = st.session_state.modelo.predict(X_novo)[0]
            st.success(f"🧠 Previsão automática: **{pred}**")

            # Mostrar exemplos semelhantes à categoria
            df = st.session_state.keywords
            exemplos = df[df["categoria"] == pred]["texto"].tolist()
            if exemplos:
                st.markdown(f"**📚 Exemplos que ajudaram nessa previsão ({pred}):**")
                st.write(exemplos[:3])
        else:
            st.info("✏️ Insira texto ou imagem para prever.")
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

