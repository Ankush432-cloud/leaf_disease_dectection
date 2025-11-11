import os
import streamlit as st
import requests
import json
import time
import random
from typing import Optional, List, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.language_models.llms import LLM

# Optional inline API key fallback (paste here if you want to hardcode)
# NOTE: This will only take effect if MISTRAL_API_KEY is not already set in the environment.
MISTRAL_API_KEY_INLINE = "w3PxX1EiQt10pySK0rP08HFSSiYnQKH5"

if MISTRAL_API_KEY_INLINE and not os.environ.get("MISTRAL_API_KEY"):
    os.environ["MISTRAL_API_KEY"] = MISTRAL_API_KEY_INLINE

def get_mistral_api_key() -> Optional[str]:
    """Read Mistral API key from common places and set env var.

    Order: st.session_state.mistral_api_key -> st.secrets -> os.environ.
    Sets MISTRAL_API_KEY in env for downstream libs.
    """
    # session_state (if previously saved elsewhere in the app)
    key = None
    if hasattr(st, "session_state"):
        key = st.session_state.get("mistral_api_key")
        if key:
            os.environ["MISTRAL_API_KEY"] = key
            return key

    # streamlit secrets
    try:
        for name in ("MISTRAL_API_KEY", "MISTRAL_TOKEN"):
            if name in st.secrets:
                key = st.secrets[name]
                if key:
                    os.environ["MISTRAL_API_KEY"] = key
                    return key
    except Exception:
        pass

    # env var
    key = os.environ.get("MISTRAL_API_KEY")
    if key:
        os.environ["MISTRAL_API_KEY"] = key
        return key
    return None


class MistralChatRequestsLLM(LLM):
    """Simple requests-based client for Mistral chat completions API."""

    api_key: str
    model: str = "mistral-small-latest"
    temperature: float = 0.7
    max_retries: int = 4  # retries on 429/5xx

    @property
    def _llm_type(self) -> str:
        return "mistral_requests"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        url = "https://api.mistral.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "user", "content": prompt}
            ],
        }
        attempt = 0
        while True:
            try:
                resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
                if resp.status_code == 429:
                    attempt += 1
                    if attempt > self.max_retries:
                        return "Rate limited by Mistral API (429). Please wait and try again."
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after:
                        delay = float(retry_after)
                    else:
                        # exponential backoff with jitter: 1s, 2s, 4s, 8s ...
                        delay = min(8.0, 2 ** (attempt - 1)) + random.uniform(0, 0.5)
                    time.sleep(delay)
                    continue

                resp.raise_for_status()
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                if stop:
                    for s in stop:
                        if s in text:
                            text = text.split(s, 1)[0]
                return text
            except requests.exceptions.RequestException as e:
                return f"Error calling Mistral API: {e}"

@st.cache_resource
def load_chatbot():
    try:
        # Mistral API key
        api_key = get_mistral_api_key()
        if not api_key:
            st.error("❌ Mistral API key not found. Set MISTRAL_API_KEY in env or st.secrets.")
            return None

        # Load PDF (expect file in project root)
        loader = PyPDFLoader("leaf_diseases.pdf")
        docs = loader.load()
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunk_documents = text_splitter.split_documents(docs)
        
        # Create embeddings and vector store (CPU-friendly HF model)
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        db = FAISS.from_documents(chunk_documents, embedding_model)
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_template("""
        Answer the following question based only on the provided context.
        Think step by step before providing a detailed answer.
        <context>
        {context}
        </context>
        
        Question: {input}
        """)
        
        # Create LLM and chains (use Mistral cloud)
        llm = MistralChatRequestsLLM(api_key=api_key, model="mistral-small-latest", temperature=0.7)
        documents_chain = create_stuff_documents_chain(llm, prompt)
        retriever = db.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, documents_chain)
        
        return retrieval_chain
    except Exception as e:
        st.error(f"Error loading chatbot: {str(e)}")
        return None

def chatbot_interface():
    st.subheader("🤖 Leaf Disease Chatbot")
    st.write("Ask questions about leaf diseases and get detailed answers from the knowledge base.")
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        with st.spinner("Loading chatbot..."):
            st.session_state.chatbot = load_chatbot()
    
    if st.session_state.chatbot is None:
        st.error("❌ Chatbot failed to load. Please check if Ollama is running and the PDF file exists.")
        return
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about leaf diseases..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.chatbot.invoke({"input": prompt})
                    answer = response["answer"]
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg}) 