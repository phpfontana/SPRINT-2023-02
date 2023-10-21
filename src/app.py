import streamlit as st
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory

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
You are a helpful, respectful and honest assistant. Always answer as 
helpfully as possible, while being safe. Your answers should not include
any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain 
why instead of answering something not correct. If you don't know the answer 
to a question, please don't share false information.

Your goal is to provide answers relating to the Digital Marketing Course.

Context: {context}

Limit your response to 100 words.

Only answer questions related to Digital Marketing and nothing else.

If the question is not about Digital Marketing, do not answer it.

Only answer in Brazilian Portuguese.

If the question is in any language other than portuguese say "Desculpa, não entendi sua pergunta.".<</SYS>>
{text}[/INST]
"""

# Set prompt template
prompt_template = PromptTemplate(template=template, input_variables=["context", "text"])

# initialize the chain
chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True
    )

# Create centered main title
st.title('🦙Assistente de Marketing Digital')

# Create a text input box for the user
query = st.text_input('Em que posso ajudar?')

# If the user has entered text
if query:
    response = chain({'query': query})
    # ...and write it out to the screen
    st.write(response['result'])

    # Display source text
    with st.expander('Source Text'):
        st.write(response['source_documents'])
