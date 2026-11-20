import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from chromadb import PersistentClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

load_dotenv()

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("🤖 RAG Chatbot")
st.caption("Ask questions about the documents")

@st.cache_resource
def load_rag_chain():
    docs = [
        Document(page_content="FastAPI is a fast web framework for Python that supports async."),
        Document(page_content="Docker is a tool for containerizing applications."),
        Document(page_content="Python is a popular and simple programming language."),
        Document(page_content="LangChain is a library for building LLM-powered applications."),
    ]

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    client = PersistentClient(path="./rag_db")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        client=client
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    model = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer only based on the text below. If the answer is not in the text, say 'I don't know'.\n\n{context}"),
        ("human", "{question}")
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    return chain

rag_chain = load_rag_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask your questions..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke(prompt)
            st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})