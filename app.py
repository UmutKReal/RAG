import streamlit as st
import PyPDF2 as PDFReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer # Hugging Face tabanlı embedding hugging facein içinde zaten entegre edilmiş huggingfaace kullanmadan da(RAG) İÇİN kullanılabiliyor
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

HFmodel = "intfloat/multilingual-e5-large-instruct"

# pdf içeriğimizi alıp hepsini tek bir stringe çevirmek
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PDFReader.PdfReader(pdf)
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    return text

#get_pdf_text(pdf_docs) dan dönen string değeri alıp liste halindeki chunklara ayırmak
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Chunk uzunluğu
        chunk_overlap=0,  # Çakışma miktarı (Daha iyi bağlam için artırrılabilir)
        separators=["\n\n", "\n", ".", " "]  # Öncelik sırasına göre bölme
    )
    chunks = text_splitter.split_text(text)
    return chunks

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#ileride bu kısmı cloud storage yapısına çevireceğim
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#ELDE ETTİĞİMİZ CHUNKLARI vektör databseleirn ekelem eişini yapacağımız ksımıo
def get_vector_storage(chunks, u_use_openai=False):
    if u_use_openai==True:
        embeddings = OpenAIEmbeddings()  # OpenAI tabanlı embedding
    else:
        embeddings = HuggingFaceEmbeddings(model_name=HFmodel)  # Hugging Face tabanlı embedding

    # FAISS içine chunk'ları ekle
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

def main():
    #apileri isimlendirme şeklim langchain frameworkunden sebebeldir
    load_dotenv()
    st.set_page_config(page_title="RAG", page_icon="🎈")
    st.header("RAG")
    st.text_input("Birşeyler sor")

    with st.sidebar:
        st.subheader("Dökümanlarım")
        pdfdocs = st.file_uploader("Dosya Yükle", accept_multiple_files=True)#PDFDOYA YOLLARI DONDURECEK
        if st.button("Gönder"):
            with st.spinner("Yükleniyor..."):
                # pdfin içeriğini alacağız
                raw_text = get_pdf_text(pdfdocs)
                #print(raw_text)
                # pdfi içeriğini chunklara ayıracağız
                #st.write(raw_text)
                chunks_of_text = get_text_chunks(raw_text)
                
                st.write(chunks_of_text)
                # vektör deposuna ekleyeceğiz
                vector_storage = get_vector_storage(chunks_of_text,False)  # Hugging Face kullanırken false DEAFULT OLARAK DA FALSE
            
if __name__ == '__main__':
    main()