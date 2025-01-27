from flask import Flask, request, render_template
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chatbot With Ollama"

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries"),
        ("user", "Question:{question}")
    ]
)

def generate_response(question, llm, temperature, max_tokens):
    llm = Ollama(model=llm)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question': question})
    return answer

@app.route("/", methods=["GET", "POST"])
def index():
    response = ""
    if request.method == "POST":
        question = request.form["question"]
        llm = request.form["llm"]
        temperature = float(request.form["temperature"])
        max_tokens = int(request.form["max_tokens"])
        response = generate_response(question, llm, temperature, max_tokens)
    return render_template("index.html", response=response)

if __name__ == "__main__":
    app.run(debug=True)