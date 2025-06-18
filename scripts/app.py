import streamlit as st

from response_generator import return_conversational_rag_chain,get_message_history
from streamlit import runtime
from streamlit.runtime.scriptrunner import get_script_run_ctx

def get_remote_ip() -> str:
    """Get remote ip."""
    try:
        ctx = get_script_run_ctx()
        if ctx is None:
            return None

        session_info = runtime.get_instance().get_client(ctx.session_id)
        if session_info is None:
            return None
    except Exception as e:
        return None
    return session_info.request.remote_ip


get_message_history(get_remote_ip()).clear()



st.set_page_config(page_title="Conversational AI For Tech Support", page_icon="🤖")
st.title("AI Assistant for Technical Support")
# Set up the message history
st.session_state.sessionid=get_remote_ip()

with st.sidebar:
    st.header("About")
    st.markdown(
        """
        This CAI interfaces with
        [LangChain](https://python.langchain.com/docs/get_started/introduction)
        designed to answer questions about the Stuart Turner Products.
        The agent uses  retrieval-augment generation (RAG) over both
        structured and unstructured data that has been synthetically generated.
        """
    )
# Display past messages
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if user_prompt := st.chat_input("Ask your question about Stuart Turner products here..."):
    # Add user message to chat history
    config = {"configurable": {"session_id": st.session_state.sessionid}}
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    # Display user message in chat message container
    if user_prompt in ["Create a Ticket"]:
        with st.chat_message("user"):
            st.markdown(user_prompt)
        with st.chat_message("assistant", avatar='/Users/navjeetkaur/Downloads/MemoryBot-main/images.jpeg'):
            response_answer=st.write("Sure , please provide the details")
            response= []
        with st.form("my_form"):
            st.write('Issue Description')
            name = st.text_input(label="Customer Name")
            email = st.text_input(label="Customer Email")
            phone = st.text_input(label="Customer Phone")
            option = st.selectbox( "Product Type?",("Shower Mate", "Moonsoon", "Aquaboost"),)
            problem = st.text_input(label="Issue Description")
            submit_form = st.form_submit_button(label="Submit", on_click=None)
            print(submit_form)
        # Checking if all the fields are non empty
            if submit_form:
                st.write(submit_form)
                if name and email and phone:
                # add_user_info(id, name, age, email, phone, gender)
                    st.success(
                            f"ID:  \n Name: {name}  \n Email: {email}  \n Phone: {phone}"
                        )
                    with st.chat_message("assistant", avatar='/Users/navjeetkaur/Downloads/MemoryBot-main/images.jpeg'):
                        response_answer=st.write("Here is you ticket number: 123456")
                        response= []
                else:
                    st.warning("Please fill all the fields")   
    else:
        with st.chat_message("user"):
            st.markdown(user_prompt)
        with st.chat_message("assistant", avatar='/Users/navjeetkaur/Downloads/MemoryBot-main/images.jpeg'):
                response=return_conversational_rag_chain().pick('context').invoke({"input": user_prompt}, config)
                response_answer=st.write_stream(return_conversational_rag_chain().pick("answer").stream({"input": user_prompt}, config))
            # Display assistant response in chat message container
                with st.expander("Sources" , expanded=True):
                    links=set()
                    docs=dict()
                    for doc in response:
                        if '.pdf' in doc.metadata['source']:
                            docs[doc.metadata['source']] = doc.metadata['page']
                        else:
                            links.add(doc.metadata['source'])
                    if (bool(links)):
                        st.write("Relavent Links:")
                        for i,link in enumerate(links):
                            urllink=st.write(f"{i+1}:{link}")      
                    if len(docs)!=0:
                        st.write("Relavent Documents:")
                        i=1
                        for doc,page in docs.items():
                            source_doc=st.write(f"{i}: https://www.stuart-turner.co.uk/contentfiles/{doc.split('/').pop()}")
                            i=i+1
    st.session_state.messages.append({"role": "assistant", "content": response_answer,"source":response})

    