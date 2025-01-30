import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv


load_dotenv()




prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        ("user", "Question: {question}")
    ]
)


def generate_response(question, llm, temperature, max_tokens):
    try:
        
        llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name=llm, streaming=True)
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser
        answer = chain.invoke({'question': question})
        return answer
    except Exception as e:
        st.write(f"An error occurred: {e}")
        return "Sorry, something went wrong. Please try again later."



st.title("CHATBOT MADE WITH LOVE")


llm = st.sidebar.selectbox("Select Groq Model", ["Llama3-8b-8192", "Llama3-13b-8192"])


temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)


st.write("Message here")
user_input = st.text_input("You:")


if user_input:
    response = generate_response(user_input, llm, temperature, max_tokens)
    st.write(response)
else:
    st.write("Please provide the user input.")

print(os.getenv("LANGCHAIN_API_KEY"))

