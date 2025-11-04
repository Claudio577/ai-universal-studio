import streamlit as st
from transformers import pipeline
from PIL import Image
import pandas as pd
# TfidfVectorizer substitui CountVectorizer, dando mais peso a palavras raras
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import tempfile
import os
import shutil

# ==============================
# ⚙️ Configuração inicial
# ==============================
st.set_page_config(page_title="AI Universal Studio Otimizado", page_icon="⚡", layout="wide")
st.title("⚡ AI Universal Studio - Otimizado")
st.write("Demonstração de um sistema de IA que aprende a partir de **imagens**, **textos** e **voz** (via upload) para gerar **previsões inteligentes** com modelos de maior precisão! 🚀")

# ==============================
# 🧩 Modelos (Upgrades para maior precisão)
# ==============================
@st.cache_resource
def load_caption_model():
    # Upgrade: Usando BLIP-Large para descrições de imagem mais ricas e precisas
    return pipeline("image-to-text", model="Salesforce/blip-base")

@st.cache_resource
def load_audio_model():
    # Upgrade: Usando Whisper-Medium para maior precisão na transcrição de voz
    return pipeline("automatic-speech-recognition", model="openai/whisper-medium")

@st.cache_resource
def load_translator():
    # Novo: Usando modelo de tradução da HF (Helsinki-NLP) para consistência e performance
    # Tradução de Inglês (EN) para Português (PT)
    return pipeline("translation_en_to_pt", model="Helsinki-NLP/opus-mt-en-pt")

captioner = load_caption_model()
asr = load_audio_model()
translator = load_translator()

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
# Funções utilitárias
# ======================================================
def translate_en_to_pt(text_en):
    """Realiza a tradução usando o pipeline da Hugging Face."""
    try:
        # O pipeline retorna uma lista de dicionários; pegamos o texto traduzido
        result = translator(text_en, max_length=100)
        return result[0]['translation_text']
    except Exception as e:
        st.error(f"Erro na tradução: {e}")
        return text_en # Retorna o texto original em caso de falha

def save_and_transcribe_audio(uploaded_audio):
    """Salva o áudio temporariamente e transcreve usando ASR."""
    # Cria um diretório temporário para garantir a limpeza
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, uploaded_audio.name)

    try:
        # Salva o arquivo temporariamente
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_audio.getbuffer())

        # Transcreve
        result = asr(temp_file_path)
        audio_text = result["text"]
        return audio_text
    except Exception as e:
        st.error(f"Erro durante a transcrição do áudio: {e}")
        return "Erro na transcrição"
    finally:
        # Limpa o arquivo e o diretório temporário
        shutil.rmtree(temp_dir)

# ======================================================
# 1️⃣ ETAPA 1 – BASE DE TREINAMENTO
# ======================================================
with aba[0]:
    st.header("🧩 Etapa 1 – Criar base de aprendizado (Palavras + Categorias)")
    st.write("Adicione até **3 exemplos de texto** para ensinar a IA o que significa cada categoria (Baixo, Moderado, Alto risco).")

    entradas = []
    # Permite 3 ou mais entradas
    num_inputs = st.number_input("Número de exemplos de treino:", min_value=3, max_value=10, value=3, key="num_train_examples")

    for i in range(num_inputs):
        col1, col2 = st.columns([3, 1])
        palavras = col1.text_input(f"📝 Exemplo {i+1} (texto ou frase):", key=f"texto_{i}")
        categoria = col2.selectbox(
            f"🎯 Categoria {i+1}:",
            ["Baixo", "Moderado", "Alto"],
            index=i % 3 if i < 3 else 1, # Distribui as categorias iniciais
            key=f"cat_{i}"
        )
        if palavras:
            entradas.append({"texto": palavras.strip(), "categoria": categoria})

    if entradas and st.button("💾 Salvar base de aprendizado"):
        st.session_state.keywords = [e["texto"] for e in entradas]
        st.session_state.categories = [e["categoria"] for e in entradas]
        st.success("✅ Base de aprendizado salva com sucesso!")
        st.dataframe(pd.DataFrame(entradas), use_container_width=True)

# ======================================================
# 2️⃣ ETAPA 2 – TREINAR MODELO
# ======================================================
with aba[1]:
    st.header("⚙️ Etapa 2 – Treinar modelo com base na base de aprendizado")

    if not st.session_state.keywords or not st.session_state.categories:
        st.warning("⚠️ Nenhum dado de aprendizado. Vá para a Etapa 1 primeiro.")
    else:
        st.info(f"Dados de treino disponíveis: {len(st.session_state.keywords)} exemplos.")
        if st.button("🚀 Treinar modelo agora (Usando TF-IDF)"):
            # Upgrade: TfidfVectorizer para maior precisão semântica
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(st.session_state.keywords)
            y = st.session_state.categories
            modelo = RandomForestClassifier(n_estimators=100, random_state=42)
            modelo.fit(X, y)
            st.session_state.vectorizer = vectorizer
            st.session_state.modelo = modelo
            st.success("✅ Modelo treinado com sucesso! Vá para a Etapa 3 para prever.")

        if st.session_state.modelo and st.session_state.vectorizer:
            st.info("✅ Modelo já treinado! Você pode ir para a Etapa 3.")

            # ======================================================
            # 🧠 Mostrar palavras-chave aprendidas pelo modelo
            # ======================================================
            st.subheader("🧠 Palavras-chave aprendidas pelo modelo (Vocabulário TF-IDF)")

            vocab = st.session_state.vectorizer.get_feature_names_out()
            st.write(f"Total de **{len(vocab)}** palavras aprendidas.")
            
            # Mostra o vocabulário por categoria
            st.markdown("### 📚 Palavras aprendidas por categoria:")

            tokenizer = st.session_state.vectorizer.build_analyzer()
            df_treino = pd.DataFrame({
                "texto": st.session_state.keywords,
                "categoria": st.session_state.categories
            })

            # Analisa as palavras para cada categoria
            for categoria in sorted(df_treino["categoria"].unique()):
                textos_cat = " ".join(df_treino[df_treino["categoria"] == categoria]["texto"]).lower()
                palavras_cat = set(tokenizer(textos_cat))
                st.markdown(f"**{categoria}:** " + ", ".join(sorted(palavras_cat)))

# ======================================================
# 3️⃣ ETAPA 3 – PREVISÃO (Imagem + Texto + Áudio)
# ======================================================
with aba[2]:
    st.header("🔮 Etapa 3 – Fazer previsão com novos dados (imagem + texto + áudio)")
    st.write("Envie uma **imagem**, **texto** e/ou **áudio (upload)** e clique em **Fazer previsão** para combinar as informações.")

    # Colunas para uploads
    col_img, col_audio = st.columns(2)

    with col_img:
        uploaded_img = st.file_uploader("📷 Envie uma imagem (opcional):", type=["jpg", "jpeg", "png"], key="predict_img")

    with col_audio:
        uploaded_audio = st.file_uploader("🎤 Envie um áudio de voz (opcional) - (.wav, .mp3, .m4a)", type=["wav", "mp3", "m4a"])

    # 💬 Texto opcional
    texto_input = st.text_area("💬 Texto descritivo (opcional):", key="predict_text")

    # Variáveis para armazenar o resultado da inferência
    audio_text = ""
    desc_img = ""

    # ======================================================
    # 🎤 Processamento de Áudio (Whisper-Medium)
    # ======================================================
    if uploaded_audio:
        with st.spinner("🔍 Transcrevendo áudio com Whisper-Medium..."):
            audio_text = save_and_transcribe_audio(uploaded_audio)
        if audio_text:
            st.success("✅ Transcrição concluída!")
            st.text_area("🗣️ Texto transcrito automaticamente:", value=audio_text, height=100, key="audio_text_area")
        else:
             st.warning("Falha na transcrição do áudio.")

    # ======================================================
    # 🧠 Geração da descrição da imagem (BLIP-Large + Tradução Opus-MT)
    # ======================================================
    if uploaded_img:
        image = Image.open(uploaded_img).convert("RGB")
        st.image(image, caption="📸 Imagem enviada", use_container_width=True)
        with st.spinner("🔍 Gerando descrição automática da imagem (BLIP-Large) e traduzindo..."):
            caption_en = captioner(image)[0]["generated_text"]
            desc_img = translate_en_to_pt(caption_en)

        st.info(f"Descrição em Inglês: {caption_en}")
        st.text_area("🖼️ Descrição em Português (Traduzida):", value=desc_img, height=100, key="desc_img_area")

    st.markdown("---")

    # ======================================================
    # 🧩 Análise separada de cada entrada
    # ======================================================
    st.subheader("🧩 Análise separada de cada entrada")

    if st.session_state.vectorizer and st.session_state.modelo:
        # Funções para prever
        def predict_source(text, source_name):
            if text:
                X = st.session_state.vectorizer.transform([text])
                pred = st.session_state.modelo.predict(X)[0]
                st.markdown(f"**Previsão baseada apenas no {source_name}:** 🧠 {pred}")
        
        col_pred_img, col_pred_audio, col_pred_text = st.columns(3)
        
        with col_pred_img:
            if desc_img:
                st.markdown("### 🖼️ Imagem")
                predict_source(desc_img, "Imagem")
        
        with col_pred_audio:
            if audio_text:
                st.markdown("### 🎤 Áudio")
                predict_source(audio_text, "Áudio")

        with col_pred_text:
            if texto_input:
                st.markdown("### 💬 Texto")
                predict_source(texto_input, "Texto")

    st.markdown("---")

    # ======================================================
    # 🧩 Combina todas as fontes de entrada
    # ======================================================
    entrada = f"{desc_img} {texto_input} {audio_text}".strip()
    st.text_area("🧩 Entrada combinada (Dados Multimodais):", value=entrada, height=120, key="entrada_combinada")

    # ======================================================
    # 🔑 Mostrar palavras reconhecidas pelo modelo
    # ======================================================
    if entrada and st.session_state.vectorizer:
        vocab = set(st.session_state.vectorizer.get_feature_names_out())
        tokenizer = st.session_state.vectorizer.build_analyzer()
        palavras_entrada = set(tokenizer(entrada.lower()))

        palavras_reconhecidas = palavras_entrada.intersection(vocab)
        palavras_nao_reconhecidas = palavras_entrada.difference(vocab)

        st.markdown("### 🧠 Palavras-chave reconhecidas (TF-IDF):")
        if palavras_reconhecidas:
            df_treino = pd.DataFrame({
                "texto": st.session_state.keywords,
                "categoria": st.session_state.categories
            })

            for categoria in sorted(df_treino["categoria"].unique()):
                textos_cat = " ".join(df_treino[df_treino["categoria"] == categoria]["texto"]).lower()
                palavras_cat = set(tokenizer(textos_cat))
                palavras_match = palavras_cat.intersection(palavras_reconhecidas)
                if palavras_match:
                    st.markdown(f"**{categoria}:** " + ", ".join(sorted(palavras_match)))
        else:
            st.warning("⚠️ Nenhuma palavra reconhecida do vocabulário treinado. Tente inserir palavras relacionadas à base de treino.")

        with st.expander("🔍 Palavras não reconhecidas (fora do vocabulário):"):
            if palavras_nao_reconhecidas:
                st.write(", ".join(sorted(palavras_nao_reconhecidas)))
            else:
                st.write("Nenhuma palavra fora do vocabulário.")

    # ======================================================
    # 🔍 Fazer previsão Final
    # ======================================================
    if st.button("🔍 Fazer previsão Final (Multimodal)"):
        if not st.session_state.modelo or not st.session_state.vectorizer:
            st.error("⚠️ Treine o modelo na Etapa 2 antes de fazer previsões.")
        elif not entrada.strip():
            st.error("⚠️ Insira uma imagem, texto e/ou áudio para prever.")
        else:
            X_novo = st.session_state.vectorizer.transform([entrada])
            pred = st.session_state.modelo.predict(X_novo)[0]
            cor = {"Baixo": "green", "Moderado": "orange", "Alto": "red"}[pred]

            st.markdown(
                f"<div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center; border-left: 5px solid {cor};'>"
                f"<h2>🧠 PREVISÃO FINAL DA IA: <span style='color:{cor}; font-weight: bold;'>{pred}</span></h2>"
                f"</div>",
                unsafe_allow_html=True
            )

            exemplos_relacionados = [
                kw for kw, cat in zip(st.session_state.keywords, st.session_state.categories)
                if cat == pred
            ]
            if exemplos_relacionados:
                st.markdown("📚 **Exemplos de treino que levaram a essa decisão:**")
                st.info(", ".join(exemplos_relacionados))
