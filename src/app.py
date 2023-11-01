import transformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import LlamaForCausalLM, LlamaTokenizer
from langchain.llms import HuggingFacePipeline
import torch
import streamlit as st
import utils


def main():
    # Loading model and tokenizer
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    hf_auth_token = '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'

    @st.cache_resource(show_spinner=False)
    def load_model(model_name, auth_token, cache_dir="./model/"):
        model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name,
            cache_dir=cache_dir,
            use_auth_token=auth_token,
            torch_dtype=torch.float16,
            rope_scaling={"type": "dynamic", "factor": 2},
            load_in_8bit=True
        )

        tokenizer = LlamaTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_name,
            cache_dir="./model/",
            use_auth_token=auth_token
        )

        return model, tokenizer

    model, tokenizer = load_model(model_name, hf_auth_token)

    # System prompt
    system_prompt = """
    [INST] <<SYS>>
    You are a helpful, respectful and honest assistant. 

    Always answer as helpfully as possible, while being safe.  

    Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. 

    Please ensure that your responses are socially unbiased and positive in nature. 

    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. 

    If you don't know the answer to a question, please don't share false information.

    Only answer questions related to the context and nothing else.

    Context: {context}<</SYS>>
    Question: {question}[/INST]
    """

    prompt_template = PromptTemplate(template=system_prompt, input_variables=["context", "question"])

    # Pipeline
    pipeline = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        temperature=0.0,
        max_new_tokens=256,
        repetition_penalty=1.1,
        return_full_text=True,
        max_length=4096
    )

    # LLM wrapper
    llm = HuggingFacePipeline(pipeline=pipeline)

    # Load pdf files from directory and split text into chunks
    documents = utils.load_documents()
    text_chunks = utils.split_text_into_chunks(documents)

    # Create embeddings and vectorstore
    embeddings = utils.create_embeddings()
    vector_store = utils.create_vector_store(text_chunks, embeddings)

    # Create chain
    chain_type_kwargs = {"prompt": prompt_template}
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True
    )

    # Create centered main title
    st.title('🦙Llama-2-RAG')

    # Initialize session state
    if 'user_query' not in st.session_state:
        st.session_state['user_query'] = ""

    # Function to set query based on example
    def set_query(example_query):
        st.session_state['user_query'] = example_query

    # info
    st.info("""
        **Tips on Asking Questions:**
        - Add files to the src/data/ directory to add more documents to the knowledge base.
        - Be specific and clear in your question.
        - Make sure your question is related to the provided PDF file.
        """)

    # Display example questions with buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Example 1: What can you do with Meta Spark Studio?"):
            set_query("What can you do with Meta Spark Studio?")
    with col2:
        if st.button("Example 2: Summarize Tips for designing an effect"):
            set_query("Summarize Tips for designing an effect")

    # User input
    user_query = st.text_input('How can I help you?', value=st.session_state['user_query'])
    st.session_state['user_query'] = user_query  # Update session state when user types

    # If the user has entered text
    if user_query:
        with st.spinner('Loading...'):
            response = chain({'query': user_query})
        # ...and write it out to the screen
        st.write(response['result'])

        # Display source text
        with st.expander('Source Text'):
            st.write(response['source_documents'][0])
            st.write(response['source_documents'][1])
            st.write(response['source_documents'][2])
            st.write(response['source_documents'][3])


if __name__ == '__main__':
    main()
