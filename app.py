# importing necessary flask functions
from flask import Flask, request
from flask_cors import CORS
import os
import streamlit as st
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

# Flask config
app = Flask(__name__)
CORS(app)

# Default route to test if the app is properly deployed
@app.route("/")
def read_root():
    return {"LangChainApp": "Working"}

from dotenv import load_dotenv
load_dotenv()
#openai_api_key = os.getenv('OPENAI_API_KEY')
#cohere_api_key = os.getenv('COHERE_API_KEY')
weaviate_api_key = st.secrets["WEAVIATE_API_KEY"]
weaviate_url = st.secrets["WEAVIATE_URL"]

# Retrieval code using langchain functions
import weaviate
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import Weaviate
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from prompt import PROMPT

# This route helps generate answer
@app.route("/retrieve", methods=['POST'])
def retrieve_resp():
    query = request.json.get("query")
    language = request.json.get("language", "english") 
    qa = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever(), chain_type_kwargs={"prompt": PROMPT.partial(language=language)})
    search_result = qa({"query": query})
    return {"search_result":search_result}

# This route gets list of most similar embeddings to the questions asked
@app.route("/retrieve-list", methods=['POST'])
def retrieve_list():
    query = request.json.get("query")
    k = request.json.get("k", 4)
    docs_list = vectorstore.similarity_search(query, k)
    return {"docs_list": str(docs_list)}


# Contextual compression implementation
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
# this function uses Cohere rerand method to perform Contextual Compression
def compression(k, top_n):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    compressor = CohereRerank(model='rerank-multilingual-v2.0', top_n=top_n )
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
    return compression_retriever
# This route helps generate answer using Contextual Compression
@app.route("/retrieve-compr", methods=['POST'])
def retrieve_compressed_resp():
    query = request.json.get("query")
    k = request.json.get("k", 9)
    top_n = request.json.get("top_n", 3)
    language = request.json.get("language", "english")
    compression_retriever = compression(k, top_n)
    qa = RetrievalQA.from_chain_type(llm, retriever=compression_retriever, chain_type_kwargs={"prompt": PROMPT.partial(language=language)})
    search_result = qa({"query": query})
    return {"search_result":search_result}
# This route gets list of most similar embeddings to the questions asked with Contextual Compression
@app.route("/retrieve-compr-list", methods=['POST'])
def retrieve_compressed_list():
    query = request.json.get("query")
    k = request.json.get("k", 9)
    top_n = request.json.get("top_n", 3)
    compression_retriever = compression(k, top_n)
    compressed_docs_list = compression_retriever.get_relevant_documents(query)
    return {"compressed_docs_list":str(compressed_docs_list)}

st.set_page_config(layout="centered", page_title="WikiGPT",page_icon="📖")

st.title("WikiGPT📖")
st.markdown("""  """)

with st.sidebar:
    st.subheader("""Welcome to WikiGPT, its like ChatGPT, but with Wikipedia🥸""")
    st.markdown("""Did you know @CohereAI open-sourced 94 million vectors of Wikipedia embeddings from 10 languages? And @weaviate_io
        hosted 10 million of them to search for free? With a few lines of code using @LangChainAI you can retrieve them.""")
    openai_api_key = st.text_input(label="Paste your [OpenAI API-Key](https://openai.com/product) below", type="password", placeholder="Paste your OpenAI API-Key here!")
    cohere_api_key = st.text_input(label="Paste your [Cohere API-Key](https://dashboard.cohere.ai/api-keys) below", type="password", placeholder="Paste your OpenAI API-Key here!")
    st.markdown("This app uses a public READ-ONLY Weaviate API key")
    st.markdown("""The code was forked from [@MisbahSy](https://twitter.com/MisbahSy/status/1656365356947210240?s=20). Please go check out his Twitter.""")

# Connect to the Weaviate demo databse containing 10M wikipedia vectors
# This uses a public READ-ONLY Weaviate API key
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
        retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
        compressor = CohereRerank(model='rerank-multilingual-v2.0', top_n=4 )
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
    st.error("Please tell me what language to respond in")
