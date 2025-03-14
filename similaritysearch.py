from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings  # Or any other embedding model you're using
embeddings = OllamaEmbeddings(model="llama3.2")
def similaritysearch(query):

# Initialize the same embedding model used during indexing


# Load the FAISS index
    vector_store = FAISS.load_local("ipynb_readme_index", embeddings,allow_dangerous_deserialization=True)

    results = vector_store.similarity_search(query, k=3)  # Get top 3 results
    return results
