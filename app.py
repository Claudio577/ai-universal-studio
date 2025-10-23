# ======================================================
# 1️⃣ PALAVRAS-CHAVE / CSV  → apenas gera base
# ======================================================
with aba[0]:
    st.header("🧩 Geração de Palavras-Chave e Categorias")
    st.write("""
    Adicione palavras e **associe uma categoria** a cada grupo.
    Exemplo de categorias possíveis:  
    🟢 **Baixo**, 🟡 **Moderado**, 🔴 **Alto**
    """)

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
        st.success("✅ Palavras-chave salvas para uso no treinamento!")
        st.dataframe(df)

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

            # Combina as palavras-chave com possíveis descrições de imagens (exemplo)
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

