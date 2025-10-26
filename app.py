import streamlit as st
from transformers import pipeline
from PIL import Image
from deep_translator import GoogleTranslator
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import tempfile

# ==============================
# ⚙️ Configuração inicial
# ==============================
st.set_page_config(page_title="AI Universal Studio", page_icon="🧠", layout="wide")
st.title("🧠 AI Universal Studio")
st.write("Demonstração de um sistema de IA que aprende a partir de **imagens**, **textos** e **voz** (via upload) para gerar **previsões inteligentes** ⚡")

# ==============================
# 🧩 Modelos
# ==============================
@st.cache_resource
def load_caption_model():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

@st.cache_resource
def load_audio_model():
    return pipeline("automatic-speech-recognition", model="openai/whisper-base")

captioner = load_caption_model()
asr = load_audio_model()

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
    "⚙️ Etapa 2 - Treinar Modelo",
    "🔮 Etapa 3 - Fazer Previsão"
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
# 2️⃣ ETAPA 2 – TREINAR MODELO
# ======================================================
with aba[1]:
    st.header("⚙️ Etapa 2 – Treinar modelo com base na base de aprendizado")

    if not st.session_state.keywords or not st.session_state.categories:
        st.warning("⚠️ Nenhum dado de aprendizado. Vá para a Etapa 1 primeiro.")
    else:
        if st.button("🚀 Treinar modelo agora"):
            vectorizer = CountVectorizer()
            X = vectorizer.fit_transform(st.session_state.keywords)
            y = st.session_state.categories
            modelo = RandomForestClassifier()
            modelo.fit(X, y)
            st.session_state.vectorizer = vectorizer
            st.session_state.modelo = modelo
            st.success("✅ Modelo treinado com sucesso! Vá para a Etapa 3 para prever.")

        if st.session_state.modelo and st.session_state.vectorizer:
            st.info("✅ Modelo já treinado! Você pode ir para a Etapa 3.")

            # ======================================================
            # 🧠 Mostrar palavras-chave aprendidas pelo modelo
            # ======================================================
            st.subheader("🧠 Palavras-chave aprendidas pelo modelo")

            vocab = st.session_state.vectorizer.get_feature_names_out()
            st.write(f"Total de **{len(vocab)}** palavras aprendidas.")
            st.write(", ".join(sorted(vocab)))

            # (Opcional) Mostrar por categoria
            df_treino = pd.DataFrame({
                "texto": st.session_state.keywords,
                "categoria": st.session_state.categories
            })

            st.markdown("### 📚 Palavras aprendidas por categoria:")

df_treino = pd.DataFrame({
    "texto": st.session_state.keywords,
    "categoria": st.session_state.categories
})

for categoria in df_treino["categoria"].unique():
    textos_cat = " ".join(
        df_treino[df_treino["categoria"] == categoria]["texto"]
    ).lower()
    palavras_cat = set(st.session_state.vectorizer.build_tokenizer()(textos_cat))
    st.markdown(f"**{categoria}:** " + ", ".join(sorted(palavras_cat)))


# ======================================================
# 3️⃣ ETAPA 3 – PREVISÃO (Imagem + Texto + Áudio)
# ======================================================
with aba[2]:
    st.header("🔮 Etapa 3 – Fazer previsão com novos dados (imagem + texto + áudio)")
    st.write("Envie uma **imagem**, **texto** e/ou **áudio (upload)** e clique em **Fazer previsão** para combinar as informações.")

    # 📷 Imagem opcional
    uploaded_img = st.file_uploader("📷 Envie uma imagem (opcional):", type=["jpg", "jpeg", "png"], key="predict_img")

    # 💬 Texto opcional
    texto_input = st.text_area("💬 Texto descritivo (opcional):", key="predict_text")

    # 🎤 Upload de áudio (opcional)
    st.subheader("🎤 Envie um áudio de voz (opcional)")
    uploaded_audio = st.file_uploader("🎧 Arquivo de áudio (.wav, .mp3, .m4a)", type=["wav", "mp3", "m4a"])

    audio_text = ""
    if uploaded_audio:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_audio.read())
            tmp_path = tmp.name
        with st.spinner("🔍 Transcrevendo áudio..."):
            result = asr(tmp_path)
            audio_text = result["text"]
        st.success("✅ Transcrição concluída!")
        st.text_area("🗣️ Texto transcrito automaticamente:", value=audio_text, height=100, key="audio_text_area")

    # 🧠 Geração da descrição da imagem
    desc_img = ""
    if uploaded_img:
        image = Image.open(uploaded_img).convert("RGB")
        st.image(image, caption="📸 Imagem enviada", use_container_width=True)
        with st.spinner("🔍 Gerando descrição automática da imagem..."):
            caption_en = captioner(image)[0]["generated_text"]
            desc_img = GoogleTranslator(source="en", target="pt").translate(caption_en)

    # 🧩 Combina todas as fontes de entrada
    entrada = f"{desc_img} {texto_input} {audio_text}".strip()
    st.text_area("🧩 Entrada combinada:", value=entrada, height=120)

    # 🧩 Combina todas as fontes de entrada
entrada = f"{desc_img} {texto_input} {audio_text}".strip()
st.text_area("🧩 Entrada combinada:", value=entrada, height=120, key="entrada_combinada")

# ======================================================
# 🔑 Mostrar palavras reconhecidas pelo modelo
# ======================================================
if entrada and st.session_state.vectorizer:
    vocab = set(st.session_state.vectorizer.get_feature_names_out())
    palavras_entrada = set(st.session_state.vectorizer.build_tokenizer()(entrada.lower()))
    
    palavras_reconhecidas = palavras_entrada.intersection(vocab)
    palavras_nao_reconhecidas = palavras_entrada.difference(vocab)

    st.markdown("### 🧠 Palavras reconhecidas pelo modelo:")
    if palavras_reconhecidas:
        df_treino = pd.DataFrame({
    "texto": st.session_state.keywords,
    "categoria": st.session_state.categories
})

for categoria in df_treino["categoria"].unique():
    textos_cat = " ".join(df_treino[df_treino["categoria"] == categoria]["texto"]).lower()
    palavras_cat = set(st.session_state.vectorizer.build_tokenizer()(textos_cat))
    palavras_match = palavras_cat.intersection(palavras_reconhecidas)
    if palavras_match:
        st.markdown(f"**{categoria}:** " + ", ".join(sorted(palavras_match)))

    else:
        st.warning("⚠️ Nenhuma palavra reconhecida do vocabulário treinado.")

    with st.expander("🔍 Palavras não reconhecidas (fora do vocabulário):"):
        if palavras_nao_reconhecidas:
            st.write(", ".join(sorted(palavras_nao_reconhecidas)))
        else:
            st.write("Nenhuma palavra fora do vocabulário.")


    # 🔍 Fazer previsão
    if st.button("🔍 Fazer previsão"):
        if not st.session_state.modelo or not st.session_state.vectorizer:
            st.warning("⚠️ Treine o modelo na Etapa 2 antes de fazer previsões.")
        elif not entrada:
            st.warning("⚠️ Insira uma imagem, texto e/ou áudio para prever.")
        else:
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
                st.markdown("📚 **Exemplos relacionados no treino:**")
                st.write(exemplos_relacionados)
