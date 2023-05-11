# importing necessary flask functions
from flask import Flask, request
from flask_cors import CORS
import os
import streamlit as st
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

weaviate_api_key = st.secrets["WEAVIATE_API_KEY"]
weaviate_url = st.secrets["WEAVIATE_URL"]
cohere_api_key=st.secrets["COHERE_API_KEY"]

import weaviate
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import Weaviate
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

st.set_page_config(layout="centered", page_title="WikiGPT",page_icon="📖")

st.title("WikiGPT📖")

with st.expander("About this project"):
    st.subheader("""Welcome to WikiGPT, its ChatGPT, but with Wikipedia🥸""")
    st.markdown("""Unlock the power of knowledge with the WikiGPT Web App. Powered by CohereAI, Weaviate, and LangChainAI, this app gives you access to a vast collection of 94 million vectors of Wikipedia embeddings from 10 languages. Hosted by Weaviate and available for free search, these embeddings offer a wealth of information at your fingertips.
    With the WikiGPT Web App, powered by LangChainAI's advanced retrieval system, you can effortlessly explore, discover, and extract valuable insights from this vast knowledge base. Get ready to dive into the world of information with just a few clicks.
    Join us on the WikiGPT Web App and embark on an exciting journey of exploration and learning!""")



with st.sidebar:
    openai_api_key = st.text_input(label="Paste your [OpenAI API-Key](https://openai.com/product) below", type="password", placeholder="Paste your OpenAI API-Key here!")
    #cohere_api_key = st.text_input(label="Paste your [Cohere API-Key](https://dashboard.cohere.ai/api-keys) below", type="password", placeholder="Paste your OpenAI API-Key here!")
    st.markdown("This app uses a public READ-ONLY Weaviate API key")
    st.image("Xnapper-2023-05-11-08.53.41.png")
    st.markdown("""The code was forked from [@MisbahSy](https://twitter.com/MisbahSy/status/1656365356947210240?s=20). Please go check out his Twitter.""")

if cohere_api_key and openai_api_key:
    
    auth_config = weaviate.auth.AuthApiKey(api_key=weaviate_api_key) 

    client = weaviate.Client( url=weaviate_url, auth_client_secret=auth_config, 
                            additional_headers={ "X-Cohere-Api-Key": cohere_api_key})


    vectorstore = Weaviate(client,  index_name="Articles", text_key="text")
    vectorstore._query_attrs = ["text", "title", "url", "views", "lang", "_additional {distance}"]
    vectorstore.embedding =CohereEmbeddings(model="embed-multilingual-v2.0", cohere_api_key=cohere_api_key)
    llm =OpenAI(temperature=0, openai_api_key=openai_api_key)
    qa = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())
    
    language = st.text_input("What language should I respond in?", placeholder="English, French, Korean?")

    if language:
        
        query = st.text_input("Ask your questions below: ", key="input",
                                        placeholder="10 million open source vectors here, what would you like to know...")
        if query:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            compressor = CohereRerank(model='rerank-multilingual-v2.0', top_n=4)
            compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
            compressed_docs = compression_retriever.get_relevant_documents(query)

            from langchain.prompts import PromptTemplate

            prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

            {context}

            Question: {question}
            Helpful Answer in {language}:"""
            PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question","language"])

            qa = RetrievalQA.from_chain_type(llm, retriever=compression_retriever, chain_type_kwargs={"prompt": PROMPT.partial(language=language)})
            result = qa({"query": query})
            st.markdown(result['result'])
    else:
        st.error("Please tell me what language to respond in.")
else:
        st.error("Please paste your API keys in the sidebar.")
