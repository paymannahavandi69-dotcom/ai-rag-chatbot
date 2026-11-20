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

docs = [
    Document(page_content="FastAPI یک فریمورک وب سریع برای پایتون است."),
    Document(page_content="Docker یک ابزار برای کانتینری کردن اپلیکیشن‌هاست."),
    Document(page_content="پایتون یک زبان برنامه‌نویسی محبوب است."),
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
    ("system", "فقط بر اساس متن زیر جواب بده. اگه جواب توی متن نبود، بگو نمی‌دونم.\n\n{context}"),
    ("human", "{question}")
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ۸. Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

result = rag_chain.invoke("Djan go چیه؟")
print(result)