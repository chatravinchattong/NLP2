import os
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. จัดการ API Key
gemini_api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

if gemini_api_key:
    os.environ["GOOGLE_API_KEY"] = gemini_api_key

st.set_page_config(page_title="ระบบแนะนำสถานที่ท่องเที่ยว", page_icon="✈️")
st.title("✈️ ระบบแนะนำสถานที่ท่องเที่ยว")

# 2. ฟังก์ชันโหลด Vector Store
@st.cache_resource
def load_vector_store():
    docs_path = "./docs"
    if not os.path.exists(docs_path) or not os.listdir(docs_path):
        return None
    
    loader = DirectoryLoader(docs_path, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)

vector_store = load_vector_store()

# 3. ประวัติแชท
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. รับคำถาม
if user_query := st.chat_input("พิมพ์คำถามการท่องเที่ยวของคุณที่นี่..."):
    
    if not gemini_api_key:
        st.error("กรุณาตั้งค่า GEMINI_API_KEY ใน Streamlit Secrets ก่อนใช้งาน")
        st.stop()

    if vector_store is None:
        st.error("กรุณาเพิ่มไฟล์เอกสารข้อมูลท่องเที่ยว (.txt) ในโฟลเดอร์ docs ก่อนถามคำถาม")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    prompt_template = """คุณเป็นผู้ช่วยแนะนำสถานที่ท่องเที่ยวที่สุภาพ รอบรู้ และให้ข้อมูลที่แม่นยำ 
จงตอบคำถามโดยใช้ข้อมูลจาก Context ที่กำหนดให้เท่านั้น หากไม่มีข้อมูลใน Context ให้ตอบตามความจริงว่าไม่พบข้อมูลในระบบ

Context:
{context}

คำถาม: {question}
คำตอบ:"""
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # แก้ไขชื่อโมเดลเป็น gemini-2.0-flash
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=gemini_api_key,
        temperature=0.3
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    with st.chat_message("assistant"):
        with st.spinner("กำลังค้นหาข้อมูลสถานที่ท่องเที่ยว..."):
            response = rag_chain.invoke(user_query)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
