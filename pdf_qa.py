import sys

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def load_pdf(pdf_path: str):
    """Load PDF and return list of Document objects."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from PDF.")
    return documents


def split_documents(documents):
    """Split documents into chunks. Return list of chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    return chunks


def create_vectorstore(chunks):
    """Create FAISS vector store from chunks. Return vectorstore."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print(f"Created FAISS vector store with {len(chunks)} vectors.")
    return vectorstore


def format_docs(docs):
    # retriever returns a list of Documents, but the prompt wants a plain string
    return "\n\n".join(doc.page_content for doc in docs)

def build_rag_chain(vectorstore):
    """Build and return the RAG chain."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the question based only on the provided context."),
        ("user", "Context: {context}\n\nQuestion: {input}")
    ])
    llm = ChatOllama(model="gemma3:4b")
    chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def main():
    if len(sys.argv) < 2:
        print("Usage: python pdf_qa.py <path_to_pdf>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    print(f"\nLoading PDF: {pdf_path}")
    
    documents = load_pdf(pdf_path)
    chunks = split_documents(documents)
    vectorstore = create_vectorstore(chunks)
    chain = build_rag_chain(vectorstore)
    
    print("\nReady! Ask questions about your PDF (type 'quit' to exit)\n")
    
    while True:
        question = input("You: ")
        if any(word in question.lower() for word in ["quit", "exit", "q", "bye"]):
            print("Bye!")
            break
        
        print("Answer: ", end="", flush=True)
        for chunk in chain.stream(question):
            print(chunk, end="", flush=True)
        print("\n")


if __name__ == "__main__":
    main()
