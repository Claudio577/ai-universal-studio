import streamlit as st
from transformers import pipeline
from PIL import Image
from deep_translator import GoogleTranslator
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
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
# 🧭 Interface em abas
# ==============================
aba = st.tabs([
    "🖼️ Análise de Imagem",
    "💬 Análise de Texto",
    "📊 Análise de CSV / Previsões",
    "🧠 Análise Final / Previsão"
])

# ==============================
# 🔁 Sessão compartilhada entre abas
# ==============================
if "img_desc" not in st.session_state:
    st.session_state.img_desc = ""
if "txt_desc" not in st.session_state:
    st.session_state.txt_desc = ""

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

                st.session_state.img_desc = caption_pt  # 🔹 Salva a descrição para uso posterior

                resumo_opcoes = [
                    "Um toque criativo para suas redes sociais!",
                    "Perfeito para inspirar o dia ✨",
                    "Um momento simples que fala muito.",
                    "Transforme momentos em conexões 💫",
                    "Compartilhe boas vibrações 💛"
                ]
                resumo_curto = random.choice(resumo_opcoes)

                st.success("✅ Análise concluída!")
                st.markdown(f"**🇺🇸 Legenda (Inglês):** {caption_en}")
                st.markdown(f"**🇧🇷 Tradução:** {caption_pt}")
                st.markdown(f"**🪶 Resumo curto:** {resumo_curto}")

# ======================================================
# 💬 ABA 2 – Análise de Texto
# ======================================================
with aba[1]:
    st.header("💬 Análise de Texto")
    st.write("Cole ou envie um texto para análise automática com IA. O sistema irá gerar um **resumo** e **insights**.")

    texto = st.text_area("Digite ou cole seu texto aqui:", height=200, key="texto_input")

    if st.button("🧠 Analisar Texto"):
        if not texto:
            st.warning("Por favor, insira um texto para análise.")
        else:
            with st.spinner("Gerando resumo e insights..."):
                if refiner:
                    prompt = f"Resuma e destaque os pontos principais do texto: {texto}"
                    resumo = refiner(prompt, max_new_tokens=100)[0]["generated_text"]
                    st.session_state.txt_desc = resumo  # 🔹 Guarda o resumo para a aba final
                else:
                    resumo = "⚠️ Modelo de texto não carregado. Tente novamente."

            st.success("✅ Análise concluída!")
            st.markdown(f"**📄 Texto original:** {texto}")
            st.markdown(f"**🧠 Resumo automático:** {resumo}")

# ======================================================
# 📊 ABA 3 – Análise de CSV / Previsões (versão melhorada)
# ======================================================
with aba[2]:
    st.header("📊 Análise e Previsão Automática com CSV")
    st.write("""
    Envie uma planilha **.csv** com seus dados.  
    O sistema detecta automaticamente colunas numéricas e textuais, treina um modelo preditivo e mostra o desempenho.
    """)

    uploaded_csv = st.file_uploader("📎 Envie o arquivo CSV", type=["csv"], key="csv")

    if uploaded_csv:
        df = pd.read_csv(uploaded_csv)

        st.subheader("📋 Visualização dos dados")
        st.dataframe(df.head())

        # Selecionar índice (opcional)
        indice = st.selectbox("Selecione uma coluna de índice (opcional):", ["(nenhuma)"] + list(df.columns))
        if indice != "(nenhuma)":
            df = df.set_index(indice)

        st.write(f"🔢 Total de linhas: {len(df)}, colunas: {len(df.columns)}")

        # Escolher coluna alvo
        colunas = list(df.columns)
        target_col = st.selectbox("🎯 Escolha a coluna de resultado (target):", ["(nenhuma)"] + colunas)

        if target_col != "(nenhuma)":
            st.divider()
            st.subheader("⚙️ Treinando modelo...")

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

            # Importância das variáveis
            importancias = pd.Series(modelo.feature_importances_, index=X.columns).sort_values(ascending=False)
            st.subheader("📈 Importância das Variáveis (Top 10)")
            fig, ax = plt.subplots()
            importancias.head(10).plot(kind='barh', ax=ax)
            st.pyplot(fig)

            # 🔮 Previsão com novo caso
            st.divider()
            st.subheader("🔮 Fazer uma nova previsão")

            entrada = {}
            for col in X.columns:
                entrada[col] = st.text_input(f"{col}:", "")

            if st.button("Prever Resultado"):
                entrada_df = pd.DataFrame([entrada])
                entrada_df = entrada_df.reindex(columns=X.columns, fill_value=0)
                pred = modelo.predict(entrada_df)[0]
                st.info(f"🧠 Resultado previsto: **{pred}**")

                # Mostrar contribuições das variáveis
                st.caption("O modelo levou em conta as principais variáveis e padrões aprendidos nos dados.")
        else:
            st.info("Selecione a coluna de resultado (target) para treinar o modelo.")
    else:
        st.info("Envie um arquivo CSV para começar.")


# ======================================================
# 🧠 ABA 4 – Análise Final / Previsão
# ======================================================
with aba[3]:
    st.header("🧠 Análise Final / Previsão")
    st.write("Combine imagem e texto para obter uma **análise preditiva** do caso com explicação.")

    # 🔹 Campos já vêm preenchidos automaticamente das abas anteriores
    img_desc = st.text_area("📷 Descrição automática da imagem:", value=st.session_state.img_desc, height=120)
    txt_desc = st.text_area("🩺 Texto clínico ou observações:", value=st.session_state.txt_desc, height=120)

    if st.button("🔮 Gerar Análise Final"):
        combinado = (img_desc.strip() + " " + txt_desc.strip()).strip()
        if not combinado:
            st.warning("Por favor, insira descrição da imagem e/ou texto clínico.")
        else:
            with st.spinner("Gerando análise final com IA..."):
                if refiner:
                    prompt = (
                        "Analise o seguinte caso e classifique o risco como "
                        "baixo, moderado ou alto, justificando a resposta de forma profissional e ética:\n\n"
                        + combinado
                    )
                    analise = refiner(prompt, max_new_tokens=150)[0]["generated_text"]
                    st.success("✅ Análise final gerada:")
                    st.write(analise)
                else:
                    st.warning("Modelo de texto não carregado. Tente novamente.")

