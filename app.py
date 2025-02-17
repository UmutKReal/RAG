# 1. GEREKLİ KÜTÜPHANELER
import streamlit as st  # Web arayüzü için
import PyPDF2 as PDFReader  # PDF okuma için
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Metin bölme
from langchain_community.embeddings import OpenAIEmbeddings  # OpenAI embedding'leri
from langchain_community.vectorstores import FAISS  # Vektör veritabanı
from dotenv import load_dotenv  # Çevresel değişkenler (.env)
from langchain.memory import ConversationBufferMemory  # Sohbet hafızası
from langchain.chains import ConversationalRetrievalChain  # Sohbet zinciri
from langchain_openai import ChatOpenAI  # OpenAI modeli

load_dotenv()  # .env'den API anahtarını yükle

# 2. AYARLAR
CHUNK_SIZE = 1000  # Metin parça boyutu (karakter)
CHUNK_OVERLAP = 200  # Parça çakışma miktarı
MODEL_NAME = "gpt-3.5-turbo"  # Kullanılacak AI modeli

# 3. PDF'DEN METİN ÇIKARMA
def get_pdf_text(pdf_docs):
    """PDF dosyalarından metin çıkarır"""
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PDFReader.PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:  # Boş sayfaları atla
                    text += page_text + "\n"
        except Exception as e:
            st.error(f"Hata: {pdf.name} okunamadı - {str(e)}")
    return text

# 4. METNİ PARÇALARA BÖLME
def get_text_chunks(text):
    """Metni bağlam koruyarak parçalara ayırır"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]  # Öncelikli bölme noktaları
    )
    return text_splitter.split_text(text)

# 5. VEKTÖR DEPOSU OLUŞTURMA
def get_vector_store(text_chunks):
    """Metin parçalarını vektörlere dönüştürür"""
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002"  # OpenAI'nin önerilen embedding modeli
    )
    return FAISS.from_texts(text_chunks, embeddings)  # FAISS ile vektör deposu

# 6. SOHBET ZİNCİRİ VE HAFIZA
def get_conversation_chain(vector_store):
    """AI ile konuşma zinciri oluşturur"""
    llm = ChatOpenAI(
        model_name=MODEL_NAME,
        temperature=0.3  # 0-1 arası yaratıcılık (düşük = daha faktüel)
    )
    
    memory = ConversationBufferMemory(  # Sohbet geçmişini saklar
        memory_key="chat_history",
        return_messages=True
    )
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 4}),  # En iyi 4 sonucu getir
        memory=memory,
        verbose=True  # Detaylı log
    )

# 7. STREAMLIT ARAYÜZÜ
def main():
    st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
    st.header("Belgelerinizle Sohbet 💬")
    
    # Oturum durumu yönetimi
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Kullanıcı sorusu
    user_question = st.chat_input("Belgeleriniz hakkında soru sorun:")
    
    if user_question and st.session_state.conversation:
        with st.spinner("Düşünüyorum..."):
            response = st.session_state.conversation({"question": user_question})
            st.session_state.chat_history = response["chat_history"]
            
            # Sohbet geçmişini göster
            for message in st.session_state.chat_history:
                if message.type == "human":
                    with st.chat_message("user"):
                        st.write(message.content)
                else:
                    with st.chat_message("assistant"):
                        st.write(message.content)
    
    # PDF yükleme paneli
    with st.sidebar:
        st.subheader("Dokümanlarınız")
        pdf_docs = st.file_uploader(
            "PDF'leri yükleyip 'İşle'ye tıklayın",
            accept_multiple_files=True,
            type="pdf"
        )
        
        if st.button("İşle"):
            if not pdf_docs:
                st.warning("Lütfen önce PDF yükleyin!")
                return
            
            with st.status("Dokümanlar işleniyor..."):
                # Metin çıkarma
                st.write("PDF'lerden metin çıkarılıyor...")
                raw_text = get_pdf_text(pdf_docs)
                
                if not raw_text:
                    st.error("Metin çıkarılamadı")
                    return
                
                # Metin bölme
                st.write("Metin parçalara ayrılıyor...")
                text_chunks = get_text_chunks(raw_text)
                
                # Vektör deposu
                st.write("Vektör veritabanı oluşturuluyor...")
                vector_store = get_vector_store(text_chunks)
                
                # Sohbet zinciri
                st.write("Sohbet motoru başlatılıyor...")
                st.session_state.conversation = get_conversation_chain(vector_store)
            
            st.success("Hazır! Soru sorabilirsiniz.")

if __name__ == "__main__":
    main()