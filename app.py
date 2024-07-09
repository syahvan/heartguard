import streamlit as st
from streamlit_chat import message
from streamlit_extras.let_it_rain import rain
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()  
groq_api_key = os.environ['GROQ_API_KEY']

def create_vector_db(path):
    try:
        text = []
        loader = PyPDFLoader(path)
        text.extend(loader.load())

        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100, length_function=len)
        chunks = text_splitter.split_documents(text)

        # embeddings = OllamaEmbeddings(model="all-minilm")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                                           model_kwargs={'device': 'cpu'})
        vector_db = FAISS.from_documents(chunks, embedding=embeddings)
        
        return vector_db
    except Exception as e:
        st.error(f"Failed to create vector store: {e}")
        return None


def process_question(query, vector_store):
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

        Jika tidak terdapat jawaban pada informasi tersebut, HANYA katakan saja:
        Saya tidak memiliki informasi dari pertanyaan yang diberikan. Untuk mengetahui informasi tersebut lebih lanjut, silahkan hubungi dokter.
        """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(query) + "\n\nApakah ada lagi yang ingin kamu tanyakan? 😁"
    return response

def initialize_session_state(vector_store, status, name):
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Halo! Bagaimana keadaan jantung saya? 🤔"]

    if 'generated' not in st.session_state:
        if status == 'risk':
            st.session_state['generated'] = [f"Halo, {name}!\n\nKamu memiliki resiko penyakit jantung koroner 😢\n\n" + process_question("Saya memiliki resiko penyakit jantung koroner, apa yang harus saya lakukan?", vector_store)]
        else:
            st.session_state['generated'] = [f"Halo, {name}!\n\nKamu memiliki jantung yang sehat 😊\n\n" + "Apakah ada yang ingin kamu ketahui mengenai penyakit jantung koroner?"]
            rain(
                emoji="🎉",
                font_size=18,
                falling_speed=5,
                animation_length=3,
            )

def conversation_chat(query, vector_store, history):
    response = process_question(query, vector_store)
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
    st.title("❤️‍🩹 HeartGuard")
    st.markdown("An early detection of coronary heart disease risk using an IoT and chatbot system.")

    st.subheader('Prediksi Resiko Penyakit Jantung Koroner 🫀')
    st.markdown('Penyakit jantung koroner disebut sebagai penyumbang kematian terbesar di dunia. Penyakit ini didukung oleh faktor risiko seperti kolesterol, tekanan darah tinggi, merokok, obesitas, dan diabetes. Yuk, cek risiko kamu sekarang untuk pencegahan dini dan hidup lebih sehat!')
    
    st.subheader('HeartGuard Bot 🤖')

    @st.experimental_dialog("Isi Informasi Diri Kamu 👤")
    def data_diri():
        # Nama
        name = st.text_input("Nama")
        # Jenis kelamin
        sex_option = st.radio("Jenis Kelamin", ["Laki-laki", "Perempuan"], index=None, horizontal=True)
        male = 1 if sex_option == "Laki-laki" else 0
        # Umur
        age = st.number_input("Umur", 1, 200)
        # Perokok
        perokok_option = st.radio("Apakah kamu perokok?", ["Ya", "Tidak"], index=None, horizontal=True)
        currentSmoker = 1 if perokok_option == "Ya" else 0
        cigsPerDay = st.number_input("Berapa jumlah rokok yang kamu konsumsi dalam sehari?", 0)
        # Tekanan darah
        BPmeds_option = st.radio("Apakah kamu sedang menjalani pengobatan tekanan darah?", ["Ya", "Tidak"], index=None, horizontal=True)
        BPMeds = 1 if BPmeds_option == "Ya" else 0
        # Stroke
        stroke_option = st.radio("Apakah kamu pernah mengalami stroke?", ["Ya", "Tidak"], index=None, horizontal=True)
        prevalentStroke = 1 if stroke_option == "Ya" else 0
        # Hipertensi
        hipertensi_option = st.radio("Apakah kamu pernah mengalami hipertensi?", ["Ya", "Tidak"], index=None, horizontal=True)
        prevalentHyp = 1 if hipertensi_option == "Ya" else 0
        # Diabetes
        diabetes_option = st.radio("Apakah kamu pernah mengalami diabetes?", ["Ya", "Tidak"], index=None, horizontal=True)
        diabetes = 1 if diabetes_option == "Ya" else 0
        # BMI
        berat_badan = st.number_input("Masukkan berat badan (kg)", min_value=1.0, format="%.2f")
        tinggi_badan = st.number_input("Masukkan tinggi badan (cm)", min_value=1.0, format="%.2f")
        tinggi_badan_m = tinggi_badan / 100
        BMI = round(berat_badan / (tinggi_badan_m ** 2), 1)
        if st.button("Submit"):
            #st.markdown("Setelah submit, tutup pop up dengan mengklik tanda 🗙 pada pojok kanan atas atau klik tombol ESC pada keyboard")
            st.session_state.data_diri = {"name": name, "male": male, "age": age, "currentSmoker": currentSmoker, "cigsPerDay": cigsPerDay,
                "BPMeds": BPMeds, "prevalentStroke": prevalentStroke, "prevalentHyp": prevalentHyp, "diabetes": diabetes, "BMI":BMI}
            st.rerun()

    if "data_diri" not in st.session_state:
        st.markdown('Untuk menggunakan aplikasi ini, silahkan isi form dibawah ini terlebih dahulu. Untuk membuka form klik tombol <strong>Buka Form</strong> dibawah ini.', unsafe_allow_html=True)
        if st.button("Buka Form"):
            data_diri()

    # BPM
    if "read_sensor" not in st.session_state:
        st.markdown('Selanjutnya, nyalakan perangkat IoT kamu dan masukkan jari kamu ke dalam alat agar sensor dapat membaca BPM dan kadar oksigen dalam tubuh kamu. Kemudian, klik tombol <strong>Read Sensor</strong> dibawah ini.', unsafe_allow_html=True)
        if st.button("Read Sensor"):
            bpm = 90
            spo2 = 96
            st.session_state.read_sensor = {"bpm": bpm, "spo2": spo2}
            st.rerun()
        st.markdown('Setelah selesai mengisi form dan mendapatkan data dari sensor, maka akan muncul prediksi resiko jantung kamu dan chatbot jika kamu memiliki pertanyaan seputar penyakit jantung koroner.')


    if "data_diri" in st.session_state and "read_sensor" in st.session_state:
        name = st.session_state.data_diri['name']
        BMI = st.session_state.data_diri['BMI']
        heartRate = st.session_state.read_sensor['bpm']
        spo2 = st.session_state.read_sensor['spo2']
        
        status = 'normal'

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if status == 'risk':
                st.html("""<strong>Status</strong><br><span style="font-size: 28px; color: red;">Risk</span>""")
            else:
                st.html("""<strong>Status</strong><br><span style="font-size: 28px; color: green;">Normal</span>""")
        with col2:
            if heartRate >= 60 and heartRate <= 100:
                st.html(f"""<strong>Average BPM</strong><br><span style="font-size: 28px; color: green;">{heartRate}</span>""")
            else:
                st.html(f"""<strong>Average BPM</strong><br><span style="font-size: 28px; color: red;">{heartRate}</span>""")
        with col3:
            if spo2 > 94:
                st.html(f"""<strong>Oxygen Level</strong><br><span style="font-size: 28px; color: green;">{spo2}</span>""")
            else:
                st.html(f"""<strong>Oxygen Level</strong><br><span style="font-size: 28px; color: red;">{spo2}</span>""")
        with col4:
            if BMI < 18.5:
                st.html(f"""<strong>BMI</strong><br><span style="font-size: 28px; color: red;">{BMI}</span>""")
            elif 18.5 <= BMI < 24.9:
                st.html(f"""<strong>BMI</strong><br><span style="font-size: 28px; color: green;">{BMI}</span>""")
            elif 25 <= BMI < 29.9:
                st.html(f"""<strong>BMI</strong><br><span style="font-size: 28px; color: orange;">{BMI}</span>""")
            else:
                st.html(f"""<strong>BMI</strong><br><span style="font-size: 28px; color: red;">{BMI}</span>""")

        with st.spinner('Loading chatbot...'):
            # Create vector store
            vector_store = create_vector_db("data/Penyakit Jantung Koroner.pdf")

            initialize_session_state(vector_store, status, name)
                
        display_chat_history(vector_store)

if __name__ == "__main__":
    main()

