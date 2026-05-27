import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA

CHROMA_DIR = "chroma_db"

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.environ["GROQ_API_KEY"])

qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever(search_kwargs={"k": 10}))

print("Ready. Type your question or 'quit' to exit.\n")
while True:
    q = input("You: ")
    if q.lower() == "quit":
        break
    result = qa.invoke({"query": q})
    print(f"Answer: {result['result']}\n")