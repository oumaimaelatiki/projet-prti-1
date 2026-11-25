from src.upload_pdf import index_uploaded_pdf
import streamlit as st

from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from langchain_ollama import ChatOllama

st.set_page_config(
    page_title="MaintiDoc",
    page_icon="🔧",
    layout="wide"
)

st.title("🔧 MaintiDoc")
st.subheader("Assistant documentaire intelligent")
st.write(
    "Posez vos questions sur les documents techniques "
    "et obtenez des réponses avec leurs sources."
)

# ======================================
# IMPORTATION DES DOCUMENTS PDF
# ======================================

st.divider()

st.header("📄 Importer un document PDF")

uploaded_file = st.file_uploader(
    "Sélectionnez un fichier PDF",
    type=["pdf"]
)

if uploaded_file is not None:
    st.info(f"Fichier sélectionné : {uploaded_file.name}")

    if st.button("📥 Indexer le PDF"):
        with st.spinner("Indexation du document en cours..."):
            try:
                stats = index_uploaded_pdf(uploaded_file)

                st.cache_resource.clear()

                st.success("Document indexé avec succès !")

                col1, col2, col3, col4 = st.columns(4)

                col1.metric("Pages", stats["pages"])
                col2.metric("Chunks", stats["chunks"])
                col3.metric("Images", stats["images"])
                col4.metric("Tableaux", stats["tables"])

            except Exception as e:
                st.error(f"Erreur : {e}")

st.divider()

@st.cache_resource
def load_rag():
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )

    vector_store = PGVectorStore.from_params(
        database="maintidoc_db",
        host="localhost",
        port=5433,
        user="maintidoc",
        password="maintidoc123",
        table_name="maintidoc_documents",
        embed_dim=384
    )

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model
    )

    retriever = index.as_retriever(
        similarity_top_k=4
    )

    llm = ChatOllama(
        model="qwen2.5:3b",
        temperature=0
    )

    return retriever, llm


question = st.text_input(
    "💬 Posez votre question :",
    placeholder="What is self-attention?"
)

if st.button("Rechercher", type="primary") and question.strip():

    with st.spinner("Recherche et génération en cours..."):
        try:
            retriever, llm = load_rag()

            results = retriever.retrieve(question)

            context = "\n\n".join(
                f"Source : {r.node.metadata.get('source', '?')}, "
                f"page {r.node.metadata.get('page', '?')}\n"
                f"{r.node.get_content()}"
                for r in results
            )

            prompt = f"""
Tu es MaintiDoc, un assistant documentaire.

Réponds uniquement avec le contexte fourni.
Si l'information est absente, dis-le clairement.
Cite les pages utilisées et n'invente rien.

CONTEXTE :
{context}

QUESTION :
{question}
"""

            response = llm.invoke(prompt)

            st.success("Réponse générée")
            st.markdown("### 🤖 Réponse")
            st.write(response.content)

            st.markdown("### 📚 Sources retrouvées")

            for r in results:
                metadata = r.node.metadata

                with st.expander(
                    f"Page {metadata.get('page', '?')} "
                    f"— {metadata.get('type', 'texte')}"
                ):
                    st.write(
                        f"Document : {metadata.get('source', '?')}"
                    )
                    st.write(
                        f"Score : {r.score:.3f}"
                        if r.score is not None
                        else "Score indisponible"
                    )
                    st.write(r.node.get_content())

        except Exception as e:
            st.error(f"Erreur : {e}")
