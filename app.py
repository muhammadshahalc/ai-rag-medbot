from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt   # import your system prompt

# LOAD ENV
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# INIT
app = Flask(__name__)
embeddings = download_hugging_face_embeddings()

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-chatbot"

# Connect to  Pinecone index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# LLM
llm = ChatGroq(
    temperature=0,
    groq_api_key=GROQ_API_KEY,
    model_name="llama3-70b-8192"
)


prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}\n\nRelevant context:\n{context}"),
    ]
)

@app.route("/")
def home():
    return render_template("chatbot.html")

@app.route("/get", methods=["POST"])
def chat():
    user_input = request.json.get("message")

    
    retriever = docsearch.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(user_input)

    
    context = "\n".join([doc.page_content for doc in docs])

    
    final_prompt = prompt_template.format_messages(
        question=user_input,
        context=context
    )

    
    response = llm(final_prompt)

    return jsonify({"response": response.content})


if __name__ == "__main__":
    app.run(debug=True)




