import os
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. ตั้งค่าหน้าตา Streamlit Web Application
st.set_page_config(page_title="ระบบตอบคำถาม RAG AI ผู้ช่วย", page_icon="🤖", layout="wide")

st.title("🤖 ระบบ RAG แชตบอตตอบคำถามจากคลังเอกสาร")
st.caption("ระบบแชตบอตอัจฉริยะที่ค้นหาคำตอบเฉพาะจากเอกสารความรู้ที่กำหนดเท่านั้น (รองรับทั้งภาษาไทยและอังกฤษ)")

# Sidebar สำหรับจัดการระบบและเอกสาร
with st.sidebar:
    st.header("⚙️ ตั้งค่าและจัดการระบบ")
    
    # ดึง API Key จาก Secrets ของ Streamlit (ตรงตามเงื่อนไขห้ามหลุดขึ้น Git)
    groq_api_key = st.secrets.get("GROQ_API_KEY", "")
    if not groq_api_key:
        st.error("❌ ไม่พบ GROQ_API_KEY ใน st.secrets")
        st.info("💡 เมื่อเอาขึ้น Streamlit Cloud ให้ไปที่ Advanced Settings -> Secrets แล้วใส่:\nGROQ_API_KEY = 'gsk_your_key'")
    else:
        st.success("✅ เชื่อมต่อ GROQ_API_KEY สำเร็จ")

    st.markdown("---")
    st.subheader("📁 ข้อมูลเอกสารในคลัง")
    
    docs_dir = "./docs"
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
        # สร้างไฟล์ตัวอย่างข้อมูล PDPA เบื้องต้นถ้ายังไม่มีไฟล์
        sample_file = os.path.join(docs_dir, "sample_pdpa.txt")
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write("""พระราชบัญญัติคุ้มครองข้อมูลส่วนบุคคล พ.ศ. 2562 (PDPA)
ข้อมูลส่วนบุคคล หมายถึง ข้อมูลเกี่ยวกับบุคคลซึ่งทำให้สามารถระบุตัวบุคคลนั้นได้ไม่ว่าทางตรงหรือทางอ้อม
สิทธิของผู้ใช้บริการตามกฎหมาย PDPA:
1. สิทธิในการขอเข้าถึงและขอรับสำเนาข้อมูลส่วนบุคคล
2. สิทธิในการขอแก้ไขข้อมูลส่วนบุคคลให้ถูกต้อง
3. สิทธิในการขอลบหรือทำลายข้อมูลส่วนบุคคล
4. สิทธิในการคัดค้านการเก็บรวบรวม ใช้ หรือเปิดเผยข้อมูล

เงื่อนไขการเปิดเผยข้อมูล:
การเปิดเผยข้อมูลส่วนบุคคลต้องได้รับความยินยอมจากเจ้าของข้อมูลส่วนบุคคลก่อน เว้นแต่เป็นไปตามข้อยกเว้นทางกฎหมาย""")

    files = os.listdir(docs_dir)
    if files:
        st.write("รายการไฟล์ในโฟลเดอร์ `./docs`:")
        for file in files:
            st.text(f"📄 {file}")
    else:
        st.warning("ยังไม่มีไฟล์เอกสาร กรุณาเพิ่มไฟล์ลงในโฟลเดอร์ ./docs")

# 2. ฟังก์ชันสำหรับ Document Loading & Chunking & Vector Search (FAISS)
@st.cache_resource
def setup_vector_store():
    if not os.path.exists(docs_dir) or not os.listdir(docs_dir):
        return None

    # Step 1: Document Loading
    documents = []
    
    # โหลดไฟล์ .txt
    txt_loader = DirectoryLoader(docs_dir, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    documents.extend(txt_loader.load())
    
    # โหลดไฟล์ .pdf
    pdf_loader = DirectoryLoader(docs_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents.extend(pdf_loader.load())

    if not documents:
        return None

    # Step 2: Document Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)

    # Step 3: Embedding & Vector Search (FAISS)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    return vector_store

# สร้าง/โหลด Vector Store
with st.spinner("กำลังเตรียมคลังข้อมูลและประมวลผล Vector Database..."):
    vector_store = setup_vector_store()

# 3. Prompt Engineering (ตรงตามเงื่อนไข: ตอบจาก Context เท่านั้น ถ้าไม่มีให้ตอบว่า "ไม่พบข้อมูล")
system_prompt_template = """คุณคือผู้ช่วยตอบคำถามอัจฉริยะ
จงตอบคำถามโดยอ้างอิงจากข้อมูลบริบท (Context) ที่กำหนดให้เท่านั้น 
หากใน Context ไม่มีข้อมูลที่สามารถตอบคำถามได้ ให้ตอบว่า "ไม่พบข้อมูล" โดยไม่ต้องพยายามคาดเดาหรือสร้างคำตอบขึ้นมาเอง

บริบท (Context):
{context}

คำถาม:
{question}

คำตอบ:"""

prompt = ChatPromptTemplate.from_template(system_prompt_template)

# 4. Chatbot Interface & Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# แสดงประวัติการสนทนา
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 เอกสารอ้างอิงที่ใช้ (Source Context)"):
                for idx, src in enumerate(message["sources"], 1):
                    st.write(f"**ชิ้นที่ {idx}** (จากไฟล์: `{src['source']}`):")
                    st.caption(src["text"])

# ช่องรับคำถามจากผู้ใช้
if user_query := st.chat_input("พิมพ์คำถามของคุณที่นี่ (เช่น สิทธิของผู้ใช้ตาม PDPA มีอะไรบ้าง)..."):
    if not groq_api_key:
        st.error("กรุณาตั้งค่า GROQ_API_KEY ก่อนใช้งาน")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    if vector_store is None:
        st.error("กรุณาเพิ่มไฟล์เอกสารในโฟลเดอร์ docs ก่อนถามคำถาม")
        st.stop()

    # Step 5: Retrieval & Large Language Model Call
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    retrieved_docs = retriever.invoke(user_query)

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.1
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": lambda x: format_docs(retrieved_docs), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    with st.chat_message("assistant"):
        with st.spinner("กำลังค้นหาข้อมูลและสรุปคำตอบ..."):
            response = rag_chain.invoke(user_query)
            st.markdown(response)

            # แสดงเอกสารอ้างอิงที่ใช้ตอบทุกครั้ง
            sources = []
            if retrieved_docs:
                with st.expander("📚 เอกสารอ้างอิงที่ใช้ (Source Context)"):
                    for idx, doc in enumerate(retrieved_docs, 1):
                        source_file = doc.metadata.get("source", "เอกสารในระบบ")
                        sources.append({"source": source_file, "text": doc.page_content})
                        st.write(f"**ชิ้นที่ {idx}** (จากไฟล์: `{source_file}`):")
                        st.caption(doc.page_content)

    # บันทึกประวัติ
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": sources
    })