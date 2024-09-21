import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from pypdf import PdfReader

from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import os

from langchain_pinecone import PineconeEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter
import time
import asyncio
from langchain.chains import RetrievalQA 
from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()

index_name="tiger"

#load pdf data
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

#creating pinecone index
def create_pipecone_index(index_name):
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

   

    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=1024, 
            metric="cosine", 
            spec=ServerlessSpec(
            cloud="aws", 
            region="us-east-1"
        ) 
    ) 
        
#Chunk the content based 
def doc_chunnk(text):
    #import time
    # Chunk the document based on h2 headers.
    markdown_document = text
    headers_to_split_on = [
        ("##", "Header 2")
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )
    md_header_splits = markdown_splitter.split_text(markdown_document)
    return md_header_splits

#adding data to pinecone data base
def pinecone_add(md_header_splits,index_name):
    try:
         current_loop = asyncio.get_event_loop()
    except RuntimeError as e:
            # If no loop exists, create a new one
        current_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(current_loop)
    
    # Initialize a LangChain embedding object.
    model_name = "multilingual-e5-large"  
    embeddings = PineconeEmbeddings(  
        model=model_name,  
        pinecone_api_key=os.environ.get("PINECONE_API_KEY")  
    )  

    # Embed each chunk and upsert the embeddings into your Pinecone index.
    docsearch = PineconeVectorStore.from_documents(
        documents=md_header_splits,
        index_name=index_name,
        embedding=embeddings, 
        namespace="wondervector5000" 
    )
    time.sleep(1)
    return docsearch


#Use Pinecone’s list and query operations to look at one of the records:
def look_records(index_name):
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index = pc.Index(index_name)
    namespace = "wondervector5000"

    for ids in index.list(namespace=namespace):
        query = index.query(
            id=ids[0], 
            namespace=namespace, 
            top_k=1,
            include_values=True,
            include_metadata=True
        )
        return query

#Use the chatbot
def chatbot(user_question,docsearch):
    # Initialize a LangChain object for chatting with the LLM
    # without knowledge from Pinecone.
    llm = ChatOpenAI(
        openai_api_key=os.environ.get('OPENAI_API_KEY'),
        model_name='gpt-3.5-turbo',
        temperature=0.0
    )

    # Initialize a LangChain object for chatting with the LLM
    # with knowledge from Pinecone. 
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever()
        )

    # Define a few questions about the WonderVector5000.
    query1 = """What is main purpose of information given?"""

    query2 = """The Neural Fandango Synchronizer is giving me a 
    headache. What do I do?"""

    # Send each query to the LLM twice, first with relevant knowledge from Pincone 
    # and then without any additional knowledge.
    print("Query 1\n")
    print("Chat with knowledge:")
    print(qa.invoke(query1).get("result"))
    print("\nChat without knowledge:")
    print(llm.invoke(query1).content)
    print("\nQuery 2\n")
    print("Chat with knowledge:")
    print(qa.invoke(query2).get("result"))
    print("\nChat without knowledge:")
    print(llm.invoke(query2).content)

def get_context_retriever_chain(docsearch):
    llm = ChatOpenAI()
    
    retriever = docsearch.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain
    
def get_conversational_rag_chain(retriever_chain): 
    
    llm = ChatOpenAI()
    
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input,docsearch):
    retriever_chain = get_context_retriever_chain(docsearch)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    
    return response['answer']




# app config
st.set_page_config(page_title="PDF CHATBOT", page_icon="😶‍🌫️")
st.title("PDF chatbot")

# sidebar
with st.sidebar:
    st.header("Settings")
    pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
    if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                st.success("Done")


    


if pdf_docs is None or pdf_docs == "":
    st.info("Please enter pdf docs")

else:
    # session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you?"),
        ]

    raw_text = get_pdf_text(pdf_docs)
    create_pipecone_index(index_name)
    md_header_splits = doc_chunnk(raw_text)
    docsearch = pinecone_add(md_header_splits,index_name)
    print(docsearch)
    st.success("Done")

    print(docsearch)
    # user input
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        response = get_response(user_query,docsearch)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))
        
       

    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
  
        
