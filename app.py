import streamlit as st
import os
from dotenv import load_dotenv
from groq import Groq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Set API key from Streamlit secrets
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# Initialize Groq client
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

# Define prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to user queries."),
    ("user", "Question: {question}")
])

# Function to generate response using Groq API
def generate_response(question, model, temperature, max_tokens):
    try:
        response = groq_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": question}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        st.write(f"An error occurred: {e}")
        return "Sorry, something went wrong. Please try again later."

# Streamlit UI
st.title("CHATBOT MADE WITH LOVE")

# Select Groq model
model = st.sidebar.selectbox("Select Model", ["mixtral-8x7b", "llama3-70b", "gemma-7b"])

# Adjust response parameters
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

# User input
st.write("Message here")
user_input = st.text_input("You:")

if user_input:
    response = generate_response(user_input, model, temperature, max_tokens)
    st.write(response)
else:
    st.write("Please provide input.")
