import os
import sys
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
DB_PATH= r"D:\rahul_learning\python-project\news-nexus\data\chroma_db"

def retrieve_documents(query,k=4,keywords_filter=True):# k=4 to fetch top 4 documents
    embedding_model=OllamaEmbeddings(model="nomic-embed-text")
    vector_store = Chroma(persist_directory=DB_PATH,embedding_function=embedding_model)
    results = vector_store.similarity_search(query,k=k+2) # to get more results k+2
    final_results = []
    if keywords_filter:
        query_terms = set(query.lower().split())
        for doc,score in results:
            content = doc.page_content.lower()
            term_matches=sum(1 for term in query_terms if term in content)
            boosted_score = score-(term_matches*0.05)
            final_results.append((doc,boosted_score))
            final_results.sort(key=lambda x:x[1])
            final_results = final_results[:k]
    else:
        final_results = results[:k]
    return final_results

if __name__ == "main":
    test_query = "What is the impact of GenAI on productivity"
    retrieved_docs = retrieve_documents(test_query)
    print(f"\n---Top{len(retrieved_docs)}Results----")
    for i,(doc,score) in enumerate(retrieved_docs):
        print(f"\n [Results{i+1}] (Score: {score:.4f})")
        print(f"Source: {doc.metadata.get('source', 'unknown')}")
        print(f"Content Snippet : {doc.page_content[:200]}...")