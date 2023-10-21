import streamlit as st
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

# load pdf files from directory
loader = DirectoryLoader('src/data/', glob="*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

# split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
text_chunks = text_splitter.split_documents(documents)

# create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                   model_kwargs={'device': "cpu"})

# vectorstore
vector_store = FAISS.from_documents(text_chunks, embeddings)

# initialize the model
llm = CTransformers(
    model="TheBloke/Llama-2-13B-chat-GGUF",
    model_type="llama",
    max_new_tokens=512
)

# initialize the prompt template
template = """
[INST] <<SYS>>
Your goal is to provide answers relating to the Meta Certified Digital Marketing Associate Exam.

Always answer briefly.

Limit your response to 100 words.

Only answer questions related to Meta Certified Digital Marketing Associate Exam and nothing else.

If the question is not about Meta Certified Digital Marketing Associate Exam, do not answer it.

Context: {context}<</SYS>>
Question: {question}[/INST]
"""

# Set prompt template
prompt_template = PromptTemplate(template=template, input_variables=["context", "question"])

chain_type_kwargs = {"prompt": prompt_template}

# initialize the chain
chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True
    )

# Create centered main title
st.title('🦙Digital Marketing Assistant')

# Create a text input box for the user
query = st.text_input('How can i help you?')

# If the user has entered text
if query:
    response = chain({'query': query})
    # ...and write it out to the screen
    st.write(response['result'])

    # Display source text
    with st.expander('Source Text'):
        st.write(response['source_documents'])

