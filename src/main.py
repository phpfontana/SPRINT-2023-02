from langchain.llms import CTransformers
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Define variable to hold llama2 weights naming
model = "TheBloke/Llama-2-13B-chat-GGUF"

# initialize the model
llm = CTransformers(
    model=model,
    model_type="llama",
    max_new_tokens=256,
    callbacks=[StreamingStdOutCallbackHandler()]
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

Your goal is to provide answers relating to the Digital Marketing Course.<</SYS>>
{text}[/INST]
"""

# initialize the chain
prompt = PromptTemplate(template=template, input_variables=["text"])
llm_chain = LLMChain(prompt=prompt, llm=llm)
