import streamlit as st
from transformers import pipeline
from PIL import Image
from deep_translator import GoogleTranslator
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import io
import random

# ==============================
# ⚙️ Configuração inicial
# ==============================
st.set_page_config(page_title="AI Universal Studio", page_icon="🧠", layout="wide")

st.title("🧠 AI Universal Studio")
st.write("Um sistema de IA genérico que analisa **imagens**, **textos** e **planilhas (CSV)** para gerar **previsões automáticas** ⚡")

# ==============================
# 🧩 Carregamento de modelos
# ==============================
@st.cache_resource
def load_caption_model():
    try:
        model_name = "microsoft/git-large-coco"  # Leve e compatível com Streamlit Cloud
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
# 🧭 Interface em abas
# ==============================
aba = st.tabs(["🖼️ Análise de Imagem", "💬 Análise de Texto", "📊 Análise de CSV / Previsões"])

# ======================================================
# 🖼️ ABA 1 – Análise de Imagem
# ======================================================
with aba[0]:
    st.header("🖼️ Análise de Imagem")
    st.write("Envie uma imagem para gerar uma **descrição automática**, tradução e hashtags relacionadas.")

    uploaded_img = st.file_uploader("📤 Envie uma imagem", type=["jpg", "jpeg", "png"], key="img")

    if uploaded_img:
        image = Image.open(uploaded_img).convert("RGB")
        image = image.resize((512, 512))
        st.image(image, caption="📸 Imagem enviada", use_container_width=True)

        if st.button("✨ Gerar Análise de Imagem"):
            with st.spinner("Analisando imagem com IA..."):
                caption_en = captioner(image)[0]["generated_text"]

                # Refinar (opcional)
                if refiner:
                    prompt = f"Melhore a legenda em inglês para que soe natural e descritiva: {caption_en}"
                    caption_en = refiner(prompt, max_new_tokens=50)[0]["generated_text"]

                caption_pt = GoogleTranslator(source="en", target="pt").translate(caption_en)

                resumo_opcoes = [
                    "Um toque criativo para suas redes sociais!",
                    "Perfeito para inspirar o dia ✨",
                    "Um momento simples que fala muito.",
                    "Transforme momentos em conexões 💫",
                    "Compartilhe boas vibrações 💛"
                ]
                resumo_curto = random.choice(resumo_opcoes)

                palavras = caption_pt.lower().split()
                principais = [p.replace(",", "") for p in palavras if len(p) > 4]
                hashtags = ["#" + p for p in principais[:5]]
                hashtags_base = hashtags + ["#inspiracao", "#fotografia", "#ia", "#ai"]

            st.success("✅ Análise concluída!")
            st.markdown(f"**🇺🇸 Legenda (Inglês):** {caption_en}")
            st.markdown(f"**🇧🇷 Tradução:** {caption_pt}")
            st.markdown(f"**🪶 Resumo curto:** {resumo_curto}")
            st.markdown(f"**🏷️ Hashtags:** {' '.join(hashtags_base)}")

# ======================================================
# 💬 ABA 2 – Análise de Texto
# ======================================================
with aba[1]:
    st.header("💬 Análise de Texto")
    st.write("Cole ou envie um texto para análise automática com IA. O sistema irá gerar um **resumo** e **insights**.")

    texto = st.text_area("Digite ou cole seu texto aqui:", height=200)

    if st.button("🧠 Analisar Texto"):
        if not texto:
            st.warning("Por favor, insira um texto para análise.")
        else:
            with st.spinner("Gerando resumo e insights..."):
                if refiner:
                    prompt = f"Resuma e destaque os pontos principais do texto: {texto}"
                    resumo = refiner(prompt, max_new_tokens=100)[0]["generated_text"]
                else:
                    resumo = "⚠️ Modelo de texto não carregado. Tente novamente."

            st.success("✅ Análise concluída!")
            st.markdown(f"**📄 Texto original:** {texto}")
            st.markdown(f"**🧠 Resumo automático:** {resumo}")

# ======================================================
# 📊 ABA 3 – Análise de CSV / Previsões
# ======================================================
with aba[2]:
    st.header("📊 Análise e Previsão Automática com CSV")
    st.write("Envie uma planilha (.csv) com seus dados. Se houver uma coluna com o **resultado esperado**, o sistema treina um modelo automaticamente.")

    uploaded_csv = st.file_uploader("📎 Envie o arquivo CSV", type=["csv"], key="csv")

    if uploaded_csv:
        df = pd.read_csv(uploaded_csv)
        st.dataframe(df.head())

        colunas = list(df.columns)
        target_col = st.selectbox("Selecione a coluna de resultado (target):", ["(nenhuma)"] + colunas)

        if target_col != "(nenhuma)":
            X = df.drop(columns=[target_col])
            y = df[target_col]

            # Converter texto em números automaticamente
            X = pd.get_dummies(X)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            modelo = RandomForestClassifier()
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            st.success(f"✅ Modelo treinado com precisão de {acc*100:.2f}%")

            # Exibir importância das features
            importancias = pd.Series(modelo.feature_importances_, index=X.columns).sort_values(ascending=False)
            st.subheader("📈 Importância das Variáveis")
            fig, ax = plt.subplots()
            importancias.head(10).plot(kind='barh', ax=ax)
            st.pyplot(fig)

            # Previsão manual
            st.subheader("🔮 Fazer uma nova previsão")
            entrada = {}
            for col in X.columns:
                entrada[col] = st.text_input(f"{col}:", "")

            if st.button("Prever Resultado"):
                entrada_df = pd.DataFrame([entrada])
                entrada_df = entrada_df.reindex(columns=X.columns, fill_value=0)
                pred = modelo.predict(entrada_df)[0]
                st.info(f"🧠 Resultado previsto: **{pred}**")
        else:
            st.info("Selecione a coluna de resultado para treinar o modelo.")
    else:
        st.info("Envie um arquivo CSV para começar.")
