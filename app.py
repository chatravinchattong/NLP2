import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. ตั้งค่าหน้าเว็บ Streamlit
st.set_page_config(page_title="ระบบแนะนำการท่องเที่ยว (Travel Guide RAG)", page_icon="✈️", layout="wide")
st.title("✈️️ ระบบแนะนำการท่องเที่ยวด้วย AI (Gemini + RAG)")

# 2. ดึงค่า API Key และตั้งค่า Environment Variable
api_key = st.secrets.get("GEMINI_API_KEY", "") or st.secrets.get("GOOGLE_API_KEY", "")
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

# แสดงสถานะการเชื่อมต่อบน Sidebar
with st.sidebar:
    st.header("⚙️ ตั้งค่าและจัดการระบบ")
    if api_key:
        st.success("เชื่อมต่อ GEMINI_API_KEY สำเร็จ")
    else:
        st.error("กรุณาตั้งค่า GEMINI_API_KEY ใน Secrets บน Streamlit Cloud")

# 3. โหลด Vector Store (FAISS)
@st.cache_resource
def load_vector_store():
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        if os.path.exists("faiss_index"):
            return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        return None
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการโหลด Vector Store: {e}")
        return None

vector_store = load_vector_store()

# 4. Chatbot Interface & Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# แสดงประวัติการสนทนา
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# รับคำถามจากผู้ใช้
if user_query := st.chat_input("พิมพ์คำถามการท่องเที่ยวของคุณที่นี่ (เช่น ขอแพลนเที่ยวเชียงใหม่ 3 วัน 2 คืน)..."):
    if not api_key:
        st.error("กรุณาตั้งค่า GEMINI_API_KEY ใน Secrets ก่อนใช้งาน")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    if vector_store is None:
        st.error("กรุณาเพิ่มไฟล์เอกสารในโฟลเดอร์ docs หรือสร้าง faiss_index ก่อนถามคำถาม")
        st.stop()

    # ค้นหาข้อมูลจาก Vector Store
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # เรียกใช้งาน Google Gemini Model (ใช้อาร์กิวเมนต์ api_key)
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        api_key=api_key,
        temperature=0.2
    )

    # กำหนด Prompt Template สำหรับ RAG
    system_prompt = (
        "คุณคือผู้ช่วยแนะนำการท่องเที่ยวที่มีความรู้และเป็นมิตร "
        "โปรดตอบคำถามโดยอ้างอิงจากข้อมูลบริบท (Context) ที่กำหนดให้อย่างถูกต้องและกระชับ\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # สร้าง RAG Chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # ประมวลผลคำตอบและแสดงผล
    with st.chat_message("assistant"):
        with st.spinner("กำลังค้นหาข้อมูลและวางแผนให้คุณ..."):
            response = rag_chain.invoke({"input": user_query})
            answer = response["answer"]
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
