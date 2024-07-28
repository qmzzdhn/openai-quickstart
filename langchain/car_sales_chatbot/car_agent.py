import os

import gradio as gr
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate

from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
import pydash


def initialize_sales_bot(vector_store_dir: str = "container_agent"):
    db = FAISS.load_local(vector_store_dir, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    global SALES_BOT
    SALES_BOT = RetrievalQA.from_chain_type(llm,
                                            retriever=db.as_retriever(search_type="similarity_score_threshold",
                                                                      search_kwargs={
                                                                          "score_threshold": 0.8
                                                                      }))
    # 返回向量数据库的检索结果
    SALES_BOT.return_source_documents = True

    return SALES_BOT


def sales_chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
    # TODO: 从命令行参数中获取
    enable_chat = True

    ans = SALES_BOT({
        "query": message,
        "history": history,
    })
    # 如果检索出结果，或者开了大模型聊天模式
    # 返回 RetrievalQA combine_documents_chain 整合的结果
    if not pydash.is_empty(ans["source_documents"]) and enable_chat:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        return ans["result"]
    if pydash.is_empty(ans["source_documents"]) and enable_chat:
        template = f"""
        you as a trans-oceans trade company container agent, please refer to this chat history: 
        {history}
        and base on the knowledge of the company(may appeared in the chat history, if not, please consider yourself as a SomeAwesomeFreightCompany agent),
        give a more professional and natural response to the customer's latest question: 
        {message}
        and please remember 
        1. do not make any promising commitment to the customer as the answer may differ from official
        2. you are the freight agent, response as a freight agent with proper freight agent techniques and tones
        """
        llm = ChatOpenAI(model_name="gpt-4", temperature=0)

        prompt = PromptTemplate(template=template, input_variables=["history", "message"])
        chain = LLMChain(llm=llm, prompt=prompt)

        response = chain.run(history=history, message=message)
        print(f"[template response] {response}")
        return response
    # 否则输出套路话术
    else:
        return "这个问题我要问问领导"


def launch_gradio():
    demo = gr.ChatInterface(
        fn=sales_chat,
        title="汽车销售ChatBot",
        # retry_btn=None,
        # undo_btn=None,
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=True, server_name="localhost")


if __name__ == "__main__":
    # 初始化汽车销售机器人
    initialize_sales_bot()
    # 启动 Gradio 服务
    launch_gradio()
