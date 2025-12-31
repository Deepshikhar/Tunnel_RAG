from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama

## Data Ingestion 

## Load all the text files from the directory
dir_loader=DirectoryLoader(
    "data/pdf_files",
    glob="**/*.pdf", ## Pattern to match files  
    loader_cls= PyMuPDFLoader, ##loader class to use
    show_progress=False

)
pdf_documents=dir_loader.load()

## Splitting into chunks

def split_documents(documents,chunk_size=1000,chunk_overlap=200):
    """Split documents into smaller chunks for better RAG performance"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = text_splitter.split_documents(documents)
    return split_docs

chunks = split_documents(pdf_documents)

## Embeddings 
from utility import *
embedding_manager = EmbeddingManager()

## Create FAISS Vector Store
vectorstore = FaissVectorStore("faiss_store")
vectorstore.build_from_documents(pdf_documents)
vectorstore.load()

## Load LLM from ollama-colab

OLLAMA_URL = "https://executed-hay-rand-chains.trycloudflare.com"
llm = Ollama(
    base_url=OLLAMA_URL,
    model="llama3.1:8b"
)

## Simple RAG function: retrieve context + generate response
def rag_simple(query, retriever, llm, top_k=3):
    results = retriever.retrieve(query, top_k=top_k)

    context = "\n\n".join([doc["content"] for doc in results]) if results else ""
    if not context:
        return "No relevant context found to answer the question."

    prompt = f"""Use the following context to answer the question concisely.

                Context:
                {context}

                Question: {query}

                Answer:
            """

    response = llm.invoke(prompt)
    return response, context

rag_retriever=RAGRetriever(vectorstore,embedding_manager)

question = "what is facial recognition"
answer, context=rag_simple(question,rag_retriever,llm)
print(answer)