import os
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. ตั้งค่าหน้าตา Streamlit Web Application
st.set_page_config(page_title="ผู้ช่วยแนะนำการท่องเที่ยว", page_icon="✈️", layout="wide")

st.title("✈️ ผู้ช่วยแนะนำการท่องเที่ยว")
st.caption("ระบบแชตบอตอัจฉริยะที่ช่วยค้นหาและแนะนำข้อมูลสถานที่ท่องเที่ยว แผนการเดินทาง และรายละเอียดจากคลังเอกสาร")

# Sidebar สำหรับจัดการระบบและเอกสาร
with st.sidebar:
    st.header("⚙️ ตั้งค่าและจัดการระบบ")
    
    # ดึง API Key จาก Secrets ของ Streamlit
    gemini_api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not gemini_api_key:
        st.error("❌ ไม่พบ GEMINI_API_KEY ใน st.secrets")
        st.info("💡 เมื่อเอาขึ้น Streamlit Cloud ให้ไปที่ Advanced Settings -> Secrets แล้วใส่:\nGEMINI_API_KEY = \"AQ.Ab8RN6Iu...\"")
    else:
        st.success("✅ เชื่อมต่อ GEMINI_API_KEY สำเร็จ")

    st.markdown("---")
    st.subheader("📁 ข้อมูลเอกสารในคลัง")
    
    docs_dir = "./docs"
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
        sample_file = os.path.join(docs_dir, "sample_travel_guide.txt")
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write("""คู่มือท่องเที่ยวเชียงใหม่ 3 วัน 2 คืน

วันแรก:
- เช้า: เดินทางถึงเชียงใหม่ ไหว้พระวัดพระธาตุดอยสุเทพเพื่อเป็นสิริมงคล
- บ่าย: เที่ยวชมความงามของวัดเจดีย์หลวง และพักผ่อนที่คาเฟ่ย่านถนนนิมมานเฮมินทร์
- เย็น: เดินเที่ยวตลาดNight Bazaar หรือถนนคนเดิน (วันเสาร์-อาทิตย์) ลิ้มลองข้าวซอยไก่ต้นตำรับ

วันที่สอง:
- เช้า: เดินทางไปดอยอินทนนท์ ชมจุดสูงสุดในประเทศไทย และเดินชมเส้นทางศึกษาธรรมชาติกิ่วแม่ปาน
- บ่าย: แวะชมพระมหาธาตุนภเมทนีดลและพระมหาธาตุนภพลภูมิสิริ
- เย็น: ทานอาหารพื้นเมือง เช่น น้ำพริกหนุ่ม แคบหมู แกงฮังเล ที่ร้านอาหารแถวหางดง

วันที่สาม:
- เช้า: ซื้อของฝากที่ตลาดวโรรส (กาดหลวง) เช่น ไส้อั่ว แคบหมู ชาไทย
- บ่าย: เดินทางกลับโดยสวัสดิภาพ

ข้อแนะนำเพิ่มเติม:
- ช่วงเวลาที่น่าเที่ยวที่สุดคือ พฤศจิกายน - กุมภาพันธ์ (อากาศหนาวเย็น)
- ค่าเข้าชมดอยอินทนนท์: ผู้ใหญ่ 60 บาท, เด็ก 30 บาท""")

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

    documents = []
    txt_loader = DirectoryLoader(docs_dir, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    documents.extend(txt_loader.load())
    
    pdf_loader = DirectoryLoader(docs_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents.extend(pdf_loader.load())

    if not documents:
        return None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    return vector_store

with st.spinner("กำลังเตรียมคลังข้อมูลการท่องเที่ยวและประมวลผล Vector Database..."):
    vector_store = setup_vector_store()

# 3. Prompt Engineering
system_prompt_template = """คุณคือผู้ช่วยแนะนำการท่องเที่ยวอัจฉริยะที่มีความนอบน้อม เป็นมิตร และให้ข้อมูลถูกต้อง
จงตอบคำถามหรือให้คำแนะนำการท่องเที่ยวโดยอ้างอิงจากข้อมูลบริบท (Context) ที่กำหนดให้เท่านั้น 
หากใน Context ไม่มีข้อมูลที่สามารถตอบคำถามได้ ให้ตอบว่า "ขออภัยครับ/ค่ะ ไม่พบข้อมูลการท่องเที่ยวส่วนนี้ในคลังเอกสาร" โดยไม่ต้องพยายามคาดเดาหรือสร้างข้อมูลขึ้นมาเอง

บริบท (Context):
{context}

คำถาม:
{question}

คำตอบ:"""

prompt = ChatPromptTemplate.from_template(system_prompt_template)

# 4. Chatbot Interface & Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 เอกสารอ้างอิงข้อมูลท่องเที่ยว"):
                for idx, src in enumerate(message["sources"], 1):
                    st.write(f"**ชิ้นที่ {idx}** (จากไฟล์: `{src['source']}`):")
                    st.caption(src["text"])

if user_query := st.chat_input("พิมพ์คำถามการท่องเที่ยวของคุณที่นี่ (เช่น ขอแพลนเที่ยวเชียงใหม่ 3 วัน 2 คืน)..."):
    if not gemini_api_key:
        st.error("กรุณาตั้งค่า GEMINI_API_KEY ก่อนใช้งาน")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    if vector_store is None:
        st.error("กรุณาเพิ่มไฟล์เอกสารในโฟลเดอร์ docs ก่อนถามคำถาม")
        st.stop()

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    retrieved_docs = retriever.invoke(user_query)

    api_key = st.secrets.get("GEMINI_API_KEY", "") or st.secrets.get("GOOGLE_API_KEY", "")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    
    # เรียกใช้ Google Gemini 1.5 Flash Model
    llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=api_key,
    temperature=0.2
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
        with st.spinner("กำลังค้นหาข้อมูลและวางแผนคำตอบ..."):
            response = rag_chain.invoke(user_query)
            st.markdown(response)

            sources = []
            if retrieved_docs:
                with st.expander("📚 เอกสารอ้างอิงข้อมูลท่องเที่ยว"):
                    for idx, doc in enumerate(retrieved_docs, 1):
                        source_file = doc.metadata.get("source", "เอกสารในระบบ")
                        sources.append({"source": source_file, "text": doc.page_content})
                        st.write(f"**ชิ้นที่ {idx}** (จากไฟล์: `{source_file}`):")
                        st.caption(doc.page_content)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": sources
    })
