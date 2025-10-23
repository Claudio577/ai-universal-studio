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
st.write("Um sistema de IA genérico que analisa **imagens**, **textos** e **planilhas (CSV)** para gerar **previsões e análises inteligentes** ⚡")

# ==============================
# 🧩 Carregamento dos modelos
# ==============================
@st.cache_resource
def load_caption_model():
    try:
        model_name = "microsoft/git-large-coco"
        captioner = pipeline("image-to-text", model=model_name)
        return captioner, model_name
    except Exception:
        model_name = "Salesforce/blip-image-captioning-base"
        captioner = pipeline("image-to-text", model=model_name)
        return captioner, model_name

@st.cache_resource
def load_text_model():
    try:
        refiner = pipeline("text2text-generation", model="google/flan-t5-small")
        return refiner
    except Exception:
        return None

captioner, model_name = load_caption_model()
refiner = load_text_model()

# ==============================
# 🧭 Interface em abas (nova ordem)
# ==============================
aba = st.tabs([
    "📝 Treinar por Palavras-Chave",
    "📊 Treinar com CSV / TXT",
    "💬 Análise de Texto",
    "🖼️ Análise de Imagem",
    "🤖 Previsão Multimodal",
    "🧠 Análise Final / Explicação"
])

# ==============================
# 🔁 Sessão compartilhada
# ==============================
if "img_desc" not in st.session_state:
    st.session_state.img_desc = ""
if "txt_desc" not in st.session_state:
    st.session_state.txt_desc = ""

# ======================================================
# 📝 ABA 1 – Treinar por Palavras-Chave
# ======================================================
with aba[0]:
    st.header("📝 Treinamento Rápido por Palavras-Chave")
    st.write("""
    Digite **grupos de palavras ou frases** e associe cada grupo a uma **categoria** (por exemplo: "baixo", "moderado", "alto").  
    O sistema treina um modelo simples para usar nas previsões futuras.
    """)

    n = st.number_input("Quantos grupos de palavras você quer adicionar?", min_value=1, max_value=10, value=3)
    entradas = []
    for i in range(n):
        col1, col2 = st.columns([3, 1])
        with col1:
            palavras = st.text_input(f"Palavras/frases do grupo {i+1}:", key=f"palavras_{i}")
        with col2:
            categoria = st.text_input(f"Categoria {i+1}:", key=f"categoria_{i}")
        if palavras and categoria:
            entradas.append({"texto": palavras, "categoria": categoria})

    if len(entradas) > 1:
        df = pd.DataFrame(entradas)
        st.subheader("📋 Dados inseridos")
        st.dataframe(df)

        if st.button("🚀 Treinar modelo com essas palavras"):
            vectorizer = CountVectorizer(stop_words="portuguese")
            X = vectorizer.fit_transform(df["texto"])
            y = df["categoria"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            modelo = RandomForestClassifier()
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            st.success(f"✅ Modelo treinado com precisão de {acc*100:.2f}%")
            st.session_state.vectorizer = vectorizer
            st.session_state.modelo = modelo
            st.info("🧠 O modelo está pronto e pode ser usado na aba **🤖 Previsão Multimodal**.")

# ======================================================
# 📊 ABA 2 – Treinar com CSV / TXT
# ======================================================
with aba[1]:
    st.header("📊 Análise e Treinamento Automático com CSV / TXT")
    uploaded_csv = st.file_uploader("📎 Envie o arquivo CSV", type=["csv"])

    if uploaded_csv:
        df = pd.read_csv(uploaded_csv)
        st.dataframe(df.head())

        indice = st.selectbox("Selecione uma coluna de índice (opcional):", ["(nenhuma)"] + list(df.columns))
        if indice != "(nenhuma)":
            df = df.set_index(indice)

        colunas = list(df.columns)
        target_col = st.selectbox("🎯 Coluna de resultado (target):", ["(nenhuma)"] + colunas)

        if target_col != "(nenhuma)":
            X = df.drop(columns=[target_col])
            y = df[target_col]
            X = pd.get_dummies(X)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            modelo = RandomForestClassifier()
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.success(f"✅ Modelo treinado com precisão de {acc*100:.2f}%")

            importancias = pd.Series(modelo.feature_importances_, index=X.columns).sort_values(ascending=False)
            st.subheader("📈 Importância das Variáveis (Top 10)")
            fig, ax = plt.subplots()
            importancias.head(10).plot(kind='barh', ax=ax)
            st.pyplot(fig)

            st.session_state.modelo = modelo
            st.info("🧠 O modelo foi salvo e pode ser usado nas previsões multimodais.")
        else:
            st.info("Selecione a coluna de resultado para treinar o modelo.")

# ======================================================
# 💬 ABA 3 – Análise de Texto
# ======================================================
with aba[2]:
    st.header("💬 Análise de Texto")
    texto = st.text_area("Digite ou cole seu texto aqui:", height=200)

    if st.button("🧠 Analisar Texto"):
        if not texto:
            st.warning("Por favor, insira um texto.")
        else:
            with st.spinner("Gerando resumo..."):
                if refiner:
                    prompt = f"Resuma e destaque os pontos principais do texto: {texto}"
                    resumo = refiner(prompt, max_new_tokens=100)[0]["generated_text"]
                    st.session_state.txt_desc = resumo
                else:
                    resumo = "⚠️ Modelo de texto não carregado."
            st.success("✅ Análise concluída!")
            st.markdown(f"**📄 Texto original:** {texto}")
            st.markdown(f"**🧠 Resumo:** {resumo}")

# ======================================================
# 🖼️ ABA 4 – Análise de Imagem
# ======================================================
with aba[3]:
    st.header("🖼️ Análise de Imagem")
    uploaded_img = st.file_uploader("📤 Envie uma imagem", type=["jpg", "jpeg", "png"])
    if uploaded_img:
        image = Image.open(uploaded_img).convert("RGB")
        image = image.resize((512, 512))
        st.image(image, caption="📸 Imagem enviada", use_container_width=True)

        if st.button("✨ Gerar descrição da imagem"):
            with st.spinner("Gerando legenda..."):
                caption_en = captioner(image)[0]["generated_text"]
                caption_pt = GoogleTranslator(source="en", target="pt").translate(caption_en)
                st.session_state.img_desc = caption_pt
                st.success("✅ Descrição gerada!")
                st.write(caption_pt)

# ======================================================
# 🤖 ABA 5 – Previsão Multimodal
# ======================================================
with aba[4]:
    st.header("🤖 Previsão Multimodal (Imagem + Texto)")
    uploaded_img = st.file_uploader("📷 Envie uma imagem (opcional)", type=["jpg", "jpeg", "png"])
    texto_input = st.text_area("💬 Escreva ou cole um texto (opcional):")

    if uploaded_img or texto_input:
        desc_img = ""
        if uploaded_img:
            image = Image.open(uploaded_img).convert("RGB")
            st.image(image, caption="📸 Imagem enviada", use_container_width=True)
            caption_en = captioner(image)[0]["generated_text"]
            caption_pt = GoogleTranslator(source="en", target="pt").translate(caption_en)
            desc_img = caption_pt

        entrada_unificada = f"{desc_img} {texto_input}".strip()
        st.markdown("### 🧩 Texto combinado:")
        st.write(entrada_unificada)

        if "vectorizer" in st.session_state and "modelo" in st.session_state:
            X_novo = st.session_state.vectorizer.transform([entrada_unificada])
            pred = st.session_state.modelo.predict(X_novo)[0]
            st.success(f"🧠 Previsão automática: **{pred}**")
        else:
            st.warning("⚠️ Treine um modelo primeiro na aba de palavras-chave ou CSV.")

# ======================================================
# 🧠 ABA 6 – Análise Final / Explicação
# ======================================================
with aba[5]:
    st.header("🧠 Análise Final / Explicação")
    img_desc = st.text_area("📷 Descrição automática da imagem:", value=st.session_state.img_desc, height=120)
    txt_desc = st.text_area("🩺 Texto clínico ou observações:", value=st.session_state.txt_desc, height=120)

    if st.button("🔮 Gerar explicação final"):
        combinado = (img_desc.strip() + " " + txt_desc.strip()).strip()
        if not combinado:
            st.warning("Insira descrição da imagem e/ou texto clínico.")
        else:
            with st.spinner("Gerando análise final..."):
                if refiner:
                    prompt = (
                        "Analise o seguinte caso e classifique o risco como "
                        "baixo, moderado ou alto, justificando de forma ética e técnica:\n\n" + combinado
                    )
                    analise = refiner(prompt, max_new_tokens=150)[0]["generated_text"]
                    st.success("✅ Análise gerada:")
                    st.write(analise)
                else:
                    st.warning("Modelo de texto não carregado.")

