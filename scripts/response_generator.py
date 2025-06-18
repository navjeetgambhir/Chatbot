from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
import os
from retriever import create_retriever
from langchain.globals import set_llm_cache
from langchain_community.cache import RedisCache
import redis

load_dotenv()
# app config
REDIS_URL = "redis://localhost:6379/0"
def get_message_history(session_id: str) -> RedisChatMessageHistory:
    return RedisChatMessageHistory(session_id, url=REDIS_URL)

def return_conversational_rag_chain():
        redis_client=redis.Redis.from_url(REDIS_URL)
        set_llm_cache(RedisCache(redis_client)) 
        api_key = os.environ.get('MISTRAL_API_KEY') 
        llm = ChatMistralAI(api_key=api_key, model="mistral-small-latest",temperature=0.5,streaming=True)
        ## Contextualize question ###


        '''  messages = [
            (
                "system",
                "You are a helpful assistant that translates English to French."
                " Translate the user sentence.",
            ),
            ("human", "I love programming."),
        ]
        ai_msg = llm.invoke(messages)

        print(ai_msg.content)'''

        contextualize_q_system_prompt = (
            """Given a chat history and the latest user question which might 
            reference context in the chat history,
            formulate a standalone question which can be understood 
            without the chat history. Do NOT answer the question,
            just reformulate it if needed and otherwise return it as is"""
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [  
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            llm, create_retriever(), contextualize_q_prompt
        )
        ### Answer question ###
        system_prompt = (
            """You are Thelma a helpful technical assistant.please provide assistance 
            to the technical support team of stuart turner company to answer the customer query.
            Provide relevant answer of the following question based only on the 
            provided context,step by step.
            if the questions are out of the context provided¸ just say you are not 
            capable of answering. 
            Address to the Stuart turner website is https://www.stuart-turner.co.uk"""
            "\n\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages([("system", system_prompt),MessagesPlaceholder("chat_history"),("human", "{input}"),])
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        ### Statefully manage chat history ###

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_message_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        
        return conversational_rag_chain


#return_conversational_rag_chain()