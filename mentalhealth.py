from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import CTransformers
import streamlit as st
from htmlTemplates import css,bot_template,user_template


MODEL_PATH="./model/llama-2-7b-chat.Q8_0.gguf"
def load_moadel()-> LlamaCpp:
    """Loads Llama model"""
    callback:CallbackManager=CallbackManager([StreamingStdOutCallbackHandler()])
    n_gpu_layers=50
    n_batch=3000
    Llama_model: LlamaCpp =LlamaCpp(
        model_path=MODEL_PATH,
        temperature=0.5,
        # n_gpu_layers=n_gpu_layers,
        # n_batch=n_batch,
        max_tokens=1000,
        n_ctx=3000,
        top_p=0.5,
        callback_manager=callback,
        verbose=True
    )
    return Llama_model
llm=load_moadel()
# def load_llm():
#     llm=CTransformers(
#         model=MODEL_PATH,
#         model_type="llama",
#         max_new_tokens=512,
#         temperature=0.5
#     )
#     return llm
# llm=load_llm()
def load_model(text):
    template = """
        [INST] <<SYS>>
        you are a mental health chatbot that encourages users to express their emotions and thoughts while providing a supportive and scientific response for the given text.keep the answer short refer this previous conversation
        Question:{question}
        Answer:{answer}.
        <</SYS>>
        {text}[/INST]
        """.strip()
    prompt = PromptTemplate(template=template, input_variables=["text","question","answer"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    print(llm_chain)
    question=""
    answer=""
    if(len(st.session_state.chat_history)>0):
        question=st.session_state.chat_history[len(st.session_state.chat_history)-2]["content"]
        answer=st.session_state.chat_history[len(st.session_state.chat_history)-1]["content"]
    print(question,answer)
    ans=llm_chain.run(text=text,question=question,answer=answer)
    return ans

def handleuser_input(user_input):
    response=load_model(user_input)
    st.session_state.chat_history.append({"role":"user","content":user_input})
    st.session_state.chat_history.append({"role":"assistant","content":response})
    st.write(user_template.replace("{{MSG}}",user_input),unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}",response),unsafe_allow_html=True)

st.set_page_config(page_title="Mental Health ChatBot" , page_icon=":llama:",layout="wide")
# st.write(page_bg_image,unsafe_allow_html=True)
def main():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history=[]
    st.write(css,unsafe_allow_html=True)
    st.header("Chat with your Therapist")
    user_question=st.chat_input("feel free to ask")
    for i,message in enumerate(st.session_state.chat_history):
        if i%2 ==0:
            st.write(user_template.replace("{{MSG}}",message["content"]),unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}",message["content"]),unsafe_allow_html=True)
    if user_question:
        handleuser_input(user_question)

if __name__ =="__main__":
    main()