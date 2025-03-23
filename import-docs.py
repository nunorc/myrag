
import os, sys

# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# from langchain.schema.document import Document



# from langchain_community.vectorstores import Chroma
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores.utils import filter_complex_metadata
# from langchain_ollama import OllamaEmbeddings

# from dotenv import load_dotenv

# load_dotenv()

# def load_documents(path):
#     document_loader = PyPDFDirectoryLoader(path)
#     return document_loader.load()

# def split_documents(docs: list[Document]):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=800,
#         chunk_overlap=80,
#         length_function=len,
#         is_separator_regex=False,
#     )

#     return text_splitter.split_documents(docs)

# def calculate_chunk_ids(chunks):
#     last_page_id = None
#     current_chunk_index = 0

#     for chunk in chunks:
#         source = chunk.metadata.get("source")
#         page = chunk.metadata.get("page")
#         current_page_id = f"{source}:{page}"

#         # If the page ID is the same as the last one, increment the index.
#         if current_page_id == last_page_id:
#             current_chunk_index += 1
#         else:
#             current_chunk_index = 0

#         # Calculate the chunk ID.
#         chunk_id = f"{current_page_id}:{current_chunk_index}"
#         last_page_id = current_page_id

#         # Add it to the page meta-data.
#         chunk.metadata["id"] = chunk_id

#     return chunks

# def add_to_chroma(chunks: list[Document]):
#     embeddings = OllamaEmbeddings(model="nomic-embed-text")

#     client = chromadb.HttpClient(host="localhost", port=8000)
#     vector_store = Chroma(client=client, embedding_function=embeddings)

#     # Calculate Page IDs.
#     chunks_with_ids = calculate_chunk_ids(chunks)

#     # Add or Update the documents.
#     existing_items = vector_store.get(include=[])  # IDs are always included by default
#     existing_ids = set(existing_items["ids"])
#     print(f"Number of existing documents in DB: {len(existing_ids)}")

#     # Only add documents that don't exist in the DB.
#     new_chunks = []
#     for chunk in chunks_with_ids:
#         if chunk.metadata["id"] not in existing_ids:
#             new_chunks.append(chunk)

#     if len(new_chunks):
#         print(f"👉 Adding new documents: {len(new_chunks)}")
#         new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
#         vector_store.add_documents(new_chunks, ids=new_chunk_ids)
#         #vector_store.persist()
#     else:
#         print("✅ No new documents to add")

# def main():
#     path = sys.argv[1]
#     if not os.path.exists(path):
#         print(f"Path not found: {path}")
#         exit(0)

#     # Create (or update) the data store.
#     docs = load_documents(path)
#     chunks = split_documents(docs)
#     add_to_chroma(chunks)

from myrag import MyRAG

def main(path):
    rag = MyRAG()
    rag.import_docs(path)

if __name__ == "__main__":
    path = 'docs'
    if len(sys.argv) == 2:
        path = sys.argv[1]

    if not os.path.exists(path):
        print(f"Path not found: {path}")
        exit(0)

    main(path)
