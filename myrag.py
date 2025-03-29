
import os, logging, chromadb
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader, TextLoader
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
                 model = os.getenv('MYRAG_MODEL', 'mistral'),
                 log_level = logging.INFO
            ):
        
        # setup embeddings
        self.embeddings = OllamaEmbeddings(model=embeddings, base_url=ollama_url)

        # setup vector store
        self.client = chromadb.HttpClient(host = chromadb_host, port = chromadb_port)
        self.vector_store = Chroma(client=self.client, embedding_function=self.embeddings)

        # setup llm model
        self.model = OllamaLLM(model=model, base_url=ollama_url)

        PROMPT_TEMPLATE = """
            ### [INST] Answer the question based only on the following context:
            {context}
            ---
            ### QUESTION Answer the question based on the above context and provide the relevant information in your answer: {question}
            [/INST]
            """
        self.prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

        # logging
        logging.basicConfig(
            format = '%(asctime)s | %(levelname)s: %(message)s',
            datefmt = "%Y-%m-%d %H:%M:%S",
            level = log_level
        )
        self.logger = logging.getLogger(__name__)

        self.logger.info(f"New MyRAG, embeddings='{embeddings}' model='{model}'")

    def _load_documents(self, path):
        docs = []

        pdf_loader = PyPDFDirectoryLoader(path)
        docs += pdf_loader.load()

        txt_loader = DirectoryLoader(path, glob="**/*.txt", loader_cls=TextLoader)
        docs += txt_loader.load()

        return docs

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
            page = chunk.metadata.get("page") or 0
            current_page_id = f"{source}:{page}"

            # re-set chunk number for every page
            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0

            chunk_id = f"{current_page_id}:{current_chunk_index}"
            last_page_id = current_page_id
            chunk.metadata["id"] = chunk_id

        return chunks

    def _add_to_vector_store(self, chunks: list[Document]):
        chunks_with_ids = self._calculate_chunk_ids(chunks)

        existing_items = self.vector_store.get(include=[])
        existing_ids = set(existing_items["ids"])
        self.logger.info(f"Number of existing documents in vetor store: {len(existing_ids)}")


        new_chunks = []
        for chunk in chunks_with_ids:
            if chunk.metadata["id"] not in existing_ids:
                new_chunks.append(chunk)

        if len(new_chunks):
            self.logger.info(f"Adding new documents: {len(new_chunks)}")
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            self.vector_store.add_documents(new_chunks, ids=new_chunk_ids)
        else:
            self.logger.info("No new documents to add")

    def import_docs(self, path:str = None):
        if not os.path.exists(path):
            self.logger.error(f"Path not found: {path}")
            exit(0)

        docs = self._load_documents(path)
        chunks = self._split_documents(docs)
        self._add_to_vector_store(chunks)

    def query(self, query: str = None):
        results = self.vector_store.similarity_search_with_score(query, k=4)

        context = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt = self.prompt_template.format(context=context, question=query)

        response = self.model.invoke(prompt)

        sources = [doc.metadata.get("id", None) for doc, _score in results]
        formatted_response = f"{response}\n\nSources:\n{sources}\n\nContext:\n{context}"

        data = {
            'response': response,
            'sources': sources,
            'context': context
        }

        return formatted_response, data
