
import os, chromadb
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.schema.document import Document
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate

load_dotenv()

class MyRAG:
    def __init__(self,
                 embeddings = os.getenv('MYRAG_EMBEDDINGS', 'nomic-embed-text'),
                 ollama_url = os.getenv('MYRAG_OLLAMA_URL', 'http://localhost:11434'),
                 chromadb_host = os.getenv('MYRAG_CHROMADB_HOST', 'localhost'),
                 chromadb_port = int(os.getenv('MYRAG_CHROMADB_PORT', '8000')),
                 model = os.getenv('MYRAG_MODEL', 'mistral')
            ):
        
        # setup embeddings
        self.embeddings = OllamaEmbeddings(model=embeddings, base_url=ollama_url)

        # sestup vector store
        self.client = chromadb.HttpClient(host = chromadb_host, port = chromadb_port)
        self.vector_store = Chroma(client=self.client, embedding_function=self.embeddings)

        # setup llm model
        self.model = OllamaLLM(model=model, base_url=ollama_url)

        self.PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

    def _load_documents(self, path):
        document_loader = PyPDFDirectoryLoader(path)

        return document_loader.load()

    def _split_documents(self, docs: list[Document]):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=80,
            length_function=len,
            is_separator_regex=False,
        )

        return text_splitter.split_documents(docs)

    def _calculate_chunk_ids(self, chunks):
        last_page_id = None
        current_chunk_index = 0

        for chunk in chunks:
            source = chunk.metadata.get("source")
            page = chunk.metadata.get("page")
            current_page_id = f"{source}:{page}"

            # If the page ID is the same as the last one, increment the index.
            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0

            # calculate chunk_id
            chunk_id = f"{current_page_id}:{current_chunk_index}"
            last_page_id = current_page_id
            chunk.metadata["id"] = chunk_id

        return chunks

    def _add_to_vector_store(self, chunks: list[Document]):
        chunks_with_ids = self._calculate_chunk_ids(chunks)

        existing_items = self.vector_store.get(include=[])
        existing_ids = set(existing_items["ids"])
        print(f"Number of existing documents in vetor store: {len(existing_ids)}")

        new_chunks = []
        for chunk in chunks_with_ids:
            if chunk.metadata["id"] not in existing_ids:
                new_chunks.append(chunk)

        if len(new_chunks):
            print(f"Adding new documents: {len(new_chunks)}")
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            self.vector_store.add_documents(new_chunks, ids=new_chunk_ids)
        else:
            print("No new documents to add")

    def import_docs(self, path:str = None):
        if not os.path.exists(path):
            print(f"Path not found: {path}")
            exit(0)

        docs = self._load_documents(path)
        chunks = self._split_documents(docs)
        self._add_to_vector_store(chunks)

    def query(self, query: str = None):
        results = self.vector_store.similarity_search_with_score(query, k=5)

        context = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(self.PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context, question=query)

        response = self.model.invoke(prompt)

        sources = [doc.metadata.get("id", None) for doc, _score in results]
        formatted_response = f"{response}\n\nSources: {sources}"

        return formatted_response, { 'response': response, 'sources': sources}
