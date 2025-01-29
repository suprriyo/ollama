import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM  # Correct import

import os
from dotenv import load_dotenv
load_dotenv()

# Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chatbot With Ollama"

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user's queries."),
        ("user", "Question: {question}")
    ]
)

def generate_response(question, model_name, temperature, max_tokens):
    try:
        llm = OllamaLLM(model=model_name)  # FIXED: Using OllamaLLM instead of Ollama
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser
        answer = chain.invoke({'question': question})
        return answer
    except Exception as e:
        st.write(f"An error occurred: {e}")
        return "Sorry, something went wrong. Please try again later."


st.title("CHATBOT MADE WITH LOVE")


st.secrets["LANGCHAIN_API_KEY"]


llm_model = st.sidebar.selectbox("Select Open Source model", ["mistral"])


temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)


st.write("Message here")
user_input = st.text_input("You:")

if user_input:
    response = generate_response(user_input, llm_model, temperature, max_tokens)
    st.write(response)
else:
    st.write("Please provide user input.")
