import streamlit as st
from streamlit_chat import message
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from PyPDF2 import PdfReader
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import tempfile

load_dotenv()  
groq_api_key = os.environ['GROQ_API_KEY']


def generate_response(query, vector_store):
    # Create llm
    llm = ChatGroq(
            groq_api_key=groq_api_key, 
            model_name='llama3-70b-8192'
    )

    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an medical AI language model assistant. Your task is to generate 3
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_store.as_retriever(), llm, prompt=QUERY_PROMPT
    )

    template = """
    Pertanyaan user: {question}
    Jawablah pertanyaan tersebut menggunakan Bahasa Indonesia dan HANYA berdasarkan informasi berikut:
    {context}

    Jika tidak ada terdapat jawaban pada informasi tersebut, hanya katakan saja:
    Saya tidak memiliki informasi dari pertanyaan yang diberikan. Untuk mengetahui informasi tersebut lebih lanjut, silahkan hubungi dokter.
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(query)
    return response

def initialize_session_state(vector_store):
    status = 'risk'

    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Halo! Bagaimana keadaan jantung saya? 🤔"]

    if status == 'risk':
        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Kamu memiliki resiko penyakit jantung koroner 😢\n\n" + generate_response("Saya memiliki resiko penyakit jantung koroner, apa yang harus saya lakukan?", vector_store)]
    else:
        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Kamu memiliki jantung yang sehat 😊\n\n" + "Apakah ada yang ingin kamu ketahui mengenai penyakit jantung koroner?"]

def conversation_chat(query, vector_store, history):
    response = generate_response(query, vector_store)
    history.append((query, response))
    return response


def display_chat_history(vector_store):
    reply_container = st.container()
    container = st.container()

    with container:
        user_input = st.chat_input("Ask me something....")

        if user_input:
            with st.spinner('Generating response...'):
                output = conversation_chat(user_input, vector_store, st.session_state['history'])
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="avataaars", seed="Aneka")
                message(st.session_state["generated"][i], key=str(i), avatar_style="bottts", seed="Aneka")


def main():
    load_dotenv()  
    groq_api_key = os.environ['GROQ_API_KEY']
    st.set_page_config(page_title="Ask your Document")
    st.header("Ask your Document 💬")
    st.markdown("a Multi-Documents ChatBot App")
    
    # Create embeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # extract text from uploaded files
    pdf_reader = PdfReader("data/Penyakit Jantung Koroner.pdf")
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    text_chunks = text_splitter.split_text(text)
    
    with st.spinner('Loading...'):
        # Create vector store
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

    initialize_session_state(vector_store)
            
    display_chat_history(vector_store)

if __name__ == "__main__":
    main()

