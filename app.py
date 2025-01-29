import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set environment variables
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chatbot with Groq"

# Define the chat prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        ("user", "Question: {question}")
    ]
)

# Function to generate response using the Groq API
def generate_response(question, llm, temperature, max_tokens):
    try:
        # Initialize the ChatGroq model with the provided Groq API key and model name
        llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name=llm, streaming=True)
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser
        answer = chain.invoke({'question': question})
        return answer
    except Exception as e:
        st.write(f"An error occurred: {e}")
        return "Sorry, something went wrong. Please try again later."


# Streamlit UI for chatbot interface
st.title("CHATBOT MADE WITH LOVE")

# Sidebar for selecting the Groq model
llm = st.sidebar.selectbox("Select Groq Model", ["Llama3-8b-8192", "Llama3-13b-8192"])

# Sidebar for controlling temperature and max tokens
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

# User input field
st.write("Message here")
user_input = st.text_input("You:")

# Generate and display the response when the user inputs a question
if user_input:
    response = generate_response(user_input, llm, temperature, max_tokens)
    st.write(response)
else:
    st.write("Please provide the user input.")

# Print the API key for debugging purposes (optional)
print(os.getenv("LANGCHAIN_API_KEY"))

