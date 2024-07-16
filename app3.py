import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = None
    batch_size = 100

    for i in range(0, len(text_chunks), batch_size):
        batch_chunks = text_chunks[i:i + batch_size]
        if vector_store is None:
            vector_store = FAISS.from_texts(batch_chunks, embedding=embeddings)
        else:
            new_vector_store = FAISS.from_texts(batch_chunks, embedding=embeddings)
            vector_store.merge_from(new_vector_store)
    
    vector_store.save_local("faiss_index")
    # You are an expert assistant specializing in maritime job applications. Answer the question using only the information provided in the resumes. If the answer is not available in the resumes, clearly state, "The answer is not available in the provided resumes." Do not attempt to infer or create information outside of what is given.

def get_conversational_chain():
    prompt_template = """
    You are a specialized ATS (Applicant Tracking System) scanner with extensive knowledge of the marine industry. Evaluate the candidate's resume against the job description and provide the following:

    1. **Percentage Match:** Calculate the percentage match between the resume and the job description, considering all relevant skills, qualifications, experience, and specific industry terminology.
    2. **Missing Keywords:** Identify and list any critical keywords or phrases that are present in the job description but missing from the resume. Ensure these keywords are specific to the marine industry.
    3. **Candidate Suitability:** Provide a detailed analysis of the candidate's overall suitability for the role, highlighting key strengths, relevant experiences, and areas for improvement. Comment on how well the candidate meets the essential and desirable criteria listed in the job description.

        Your feedback should be comprehensive, precise, and focused on key aspects of the resume evaluation.

    Resume Information:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except ValueError as e:
        print(f"Error loading FAISS index: {e}")
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    print(response)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config(page_title="ESM AI-BOT")
    st.header("Chat with ESM AI-BOT")
    # user_question = st.text_input("Ask a Question from the PDF Files")
    user_question = st.text_area("Ask a Question from the PDF Files", height=300)
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
