import uuid
from typing import Tuple, List
import os
from pathlib import Path
import re
import pickle
import pandas as pd

##LANGCHAIN IMPORTS
from langchain_community.vectorstores import FAISS
from llama_parse import LlamaParse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.runnables import RunnableLambda, Runnable
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
load_dotenv()

llamaparse_api_key = os.getenv("LLAMA_PARSE_KEY")

import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def markdown_table_to_dataframe(markdown_text):
    # Regular expression to match table rows
    table_pattern = re.compile(r"\|(.+)\|\n\|[-| :]+\|\n((\|.+\|\n)+)")
    
    # Find all tables in the markdown text
    tables = table_pattern.findall(markdown_text)
    
    dataframes = []
    
    for table in tables:
        header, body = table[0], table[1]
        # print(header)
        # print(header.split('|'))
        
        # Process header
        headers = header.split('|')
        # headers = [h.strip() for h in header.split('|') if h.strip()]
        
        # Process body
        rows = []
        for row in body.strip().split("\n"):
            rows.append([r.strip() for r in row.split('|') if r.strip()])
        
        # Create DataFrame
        try:
            df = pd.DataFrame(rows, columns=headers)
            dataframes.append(df)
        except ValueError:
            continue
    
    return dataframes

class DocumentHandler:
    """Handles loading and preprocessing of documents."""

    def __init__(self, pdf_name: str, pdf_path: str, chunk_size: int = 1024, chunk_overlap: int = 100):
        """
        Initialize the DocumentHandler.

        Parameters:
        - pdf_path (str): The path to the document file.
        - chunk_size (int): Size of each text chunk.
        - chunk_overlap (int): Overlap between text chunks.
        """
        self.pdf_name = pdf_name
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Folder names
        self.main_folder = Path("data")
        self.sub_folder = self.main_folder / self.pdf_name  # Variable representing the sub-folder

        # Create main folder if it doesn't exist
        self.main_folder.mkdir(exist_ok=True)

        # Create sub-folder inside the main folder if it doesn't exist
        self.sub_folder.mkdir(exist_ok=True)

    def load_or_parse_data(self):
        data_file = f"{self.sub_folder}/parsed_data.pkl"
    
        if os.path.exists(data_file):
            # Load the parsed data from the file
            with open(data_file, "rb") as f:
                parsed_data = pickle.load(f)
        else:
            parser = LlamaParse(api_key=llamaparse_api_key, result_type="markdown")
            llama_parse_documents = parser.load_data(self.pdf_path)
            

            # Save the parsed data to a file
            with open(data_file, "wb") as f:
                pickle.dump(llama_parse_documents, f)
            
            # Set the parsed data to the variable
            parsed_data = llama_parse_documents
        
        return parsed_data

    def load_document(self) -> Tuple[List, List]:
        """
        Load and split the document into chunks.

        Returns:
        Tuple containing:
        - List of unique document chunks.
        - List of unique IDs for the document chunks.
        """
        llama_parse_documents = self.load_or_parse_data()
        
        with open(f'{self.sub_folder}/output.md', 'a') as f:  # Open the file in append mode ('a')
            for doc in llama_parse_documents:
                f.write(doc.text + '\n')
        
        loader = DirectoryLoader(self.sub_folder, glob="**/*.md", show_progress=True, loader_cls=TextLoader)
        documents = loader.load()
        # Split loaded documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        docs = text_splitter.split_documents(documents)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in docs]
        unique_ids = list(set(ids))

        # Ensure that only docs that correspond to unique ids are kept and that only one of the duplicate ids is kept
        seen_ids = set()
        unique_docs = [doc for doc, id in zip(docs, ids) if id not in seen_ids and (seen_ids.add(id) or True)]
        logger.info(f"Loaded and split document into {len(unique_docs)} unique chunks.")
        return unique_docs, unique_ids

    def setup_vector_database(self, embeddings, directory: str):
        """
        Setup the vector database.

        Parameters:
        - documents: List of document chunks.
        - ids: List of unique IDs for the document chunks.
        - embeddings: Embeddings for the documents.
        - directory (str): Directory to store the vector database.

        Returns:
        Chroma object representing the vector database.
        """
        saved_dir = os.path.join(directory, self.sub_folder)
        if os.path.exists(saved_dir):
            # Check if the directory exists and has a database file
            logger.info("Using existing vector database.")
            vectordb = FAISS.load_local(saved_dir, embeddings, 
                                        allow_dangerous_deserialization=True)
        else:
            unique_pdfdocs, unique_pdfids = self.load_document()
            logger.info("Creating new vector database.")
            vectordb = FAISS.from_documents(
                documents=unique_pdfdocs,
                embedding=embeddings
            )
            vectordb.save_local(f"{directory}/{self.sub_folder}")
            logger.info("Vector database setup completed.")
        return vectordb



class PromptManager:
    """Manages prompts for conversations."""

    def __init__(self):
        """Initialize the PromptManager."""
        self._define_prompts()

    def _define_prompts(self) -> None:
        """
        Define prompts for conversation.

        These prompts are used for generating search queries and answering user queries.
        """
        system_prompt = (
            """Use the provided context to answer the provided user query. 
            Only use the provided context to answer the query.
            Context: {context}

            If there is no provided context to answer the query, respond with "I don't know."
            If the provided context is insufficient to answer the query, also respond with "I don't know."
            """
        )
        context_q_system_prompt = (
            """
            Given the below conversation, generate a search query to look up to get information relevant to the conversation"
            """
        )
        context_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", context_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        self.context_q_prompt = context_q_prompt
        self.qa_prompt = qa_prompt
        logger.info("Prompts defined successfully.")


class LLMHandler:
    """Handler for Large Language Models."""
    def __init__(self, llm, pdf_name: str, pdf_path:str, embeddings: FastEmbedEmbeddings, 
                 chunk_size: int = 2048, chunk_overlap: int = 200, k: int = 3, 
                 score_threshold: float = 0.1, directory: str = "vectordb"):
        """
        Initialize the LLMHandler.

        Parameters:
        - llm: The large language model.
        - pdf_path: The path to the PDF file.
        - embeddings: Embeddings for the model.
        - chunk_size: Size of text chunks.
        - chunk_overlap: Overlap between text chunks.
        - k: Number of similar documents to retrieve.
        - score_threshold: Similarity score threshold.
        - directory: Directory for vector database.

        """
        self.llm = llm
        self.chat_history = []
        self.prompt_manager = PromptManager()
        self.document_handler = DocumentHandler(pdf_name, pdf_path, chunk_size, chunk_overlap)
        vectordb = self.document_handler.setup_vector_database(embeddings=embeddings, directory=directory)
        self.retriever = vectordb.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": k,
                "score_threshold": score_threshold,
            },
        )
        self.create_chain()

    # def get_history_aware_retriever(self, query_and_history):
    #     docs = create_history_aware_retriever(self.llm, self.retriever, self.prompt_manager.context_q_prompt).invoke(query_and_history)
    #     return {"docs": docs, "query_and_history": query_and_history}

    # def run_query(self, input):
    #     docs = input['docs']
    #     query_and_history = input['query_and_history']
    #     response_buffer = ""
    #     sources = []

    #     for doc in docs:
    #         dataframes = markdown_table_to_dataframe(doc.page_content)
    #         if dataframes:
    #             for i, df in enumerate(dataframes):
    #                 print(f"Table {i}:\n {df}")
    #                 agent = create_pandas_dataframe_agent(self.llm, df, agent_type="tool-calling", verbose=True, allow_dangerous_code=True)
    #                 query = query_and_history['input']

    #                 try:
    #                     agent_response = agent.invoke(query)
    #                     if 'output' in agent_response:
    #                         response_buffer += agent_response['output']
    #                     else: 
    #                         response_buffer += doc.page_content

    #                 except Exception as e:
    #                     response_buffer += doc.page_content

    #                 sources.append(doc)
    #         else:
    #             response_buffer += f"\n{doc.page_content}"
    #             sources.append(doc)

    #     query_and_history["context"] = response_buffer
    #     self.sources = sources

    #     return query_and_history

    # def generate_response(self, query_and_history):
    #     response = self.llm.invoke(query_and_history).content
    #     return {"response":response, "sources":self.sources}
    
    def create_chain(self) -> Runnable:
        """Create the retrieval chain."""
        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, self.prompt_manager.context_q_prompt)
        question_answer_chain = create_stuff_documents_chain(self.llm, self.prompt_manager.qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        self.rag_chain = rag_chain
        logger.info("Chain creation completed successfully.")
        return self.rag_chain
    # def create_chain(self):
    #     retrieve = RunnableLambda(self.get_history_aware_retriever)
    #     context = RunnableLambda(self.run_query)
    #     response = RunnableLambda(self.generate_response)
    #     chain = retrieve | context | self.prompt_manager.qa_prompt | response
    #     return chain


    def chat(self, query) -> None:
        """Initiate a chat session."""
        response_buffer = ""
        # chain = self.rag_chain.pick("answer")
        for chunk in self.rag_chain.stream({"input": query, "chat_history": self.chat_history}):
            if answer_chunk := chunk.get("answer"):
                print(f"{answer_chunk}", end="")
                response_buffer += answer_chunk
            else:
                context_chunk = chunk.get("context")
                print(f"{context_chunk}", end="")
        

        self.chat_history.extend(
            [
                HumanMessage(content=query),
                AIMessage(response_buffer),
            ]
        )
        logger.info("Chat response generated successfully.")
