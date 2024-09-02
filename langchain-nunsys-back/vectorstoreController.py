import chromadb
from langchain_chroma import Chroma
from langchain_core.runnables.base import Runnable
from langchain_openai import OpenAIEmbeddings
from uuid import uuid4
import os
from langchain_core.documents import Document
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from os import listdir
from os.path import isfile, join
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema import (
    AIMessage,
)


class VectorstoreController:
    def __init__(self, llm):
        # Initialize the vector store and persistent client
        self.db = self.initialize_db()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.llm = llm
        self.store = {}
        
    def initialize_db(self):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        persistent_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Initialize vector store with collection
        vector_store = Chroma(
            client=persistent_client,
            collection_name="documentsVec",
            embedding_function=embeddings,
        )

        return vector_store


    ## RAG Section
    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
      if session_id not in self.store:
          self.store[session_id] = ChatMessageHistory()
      return self.store[session_id]


    def query(self, text: str):
      retriever = self.db.as_retriever()
      
      system_prompt = (
          "Eres un asistente para tareas de preguntas y respuestas. "
          "Usa el siguiente contexto para responder la pregunta. "
          "Si no sabes la respuesta, di que no la sabes. "
          "Usa un máximo de tres oraciones y mantén la respuesta concisa.."
          "\n\n"
          "{context}"
      )
      

      # History Aware
      history_prompt_text = ("Dado un historial de chat y la última pregunta del usuario, "
                  " que podría hacer referencia al contexto en el historial de chat, "
                  " formula una pregunta independiente que pueda entenderse sin el historial de chat. "
                  " NO respondas la pregunta, solo reformúlala si es necesario y, si no, devuélvela tal cual")
      history_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", history_prompt_text),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
      )
      history_aware_retriever = create_history_aware_retriever(self.llm, retriever, history_prompt)
      
      qa_prompt = ChatPromptTemplate.from_messages(
          [
              ("system", system_prompt),
              MessagesPlaceholder("chat_history"),
              ("human", "{input}"),
          ]
      )

      question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
      rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

      
      # question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
      rag_chain = create_retrieval_chain(retriever, question_answer_chain)
      
      conversational_rag_chain = RunnableWithMessageHistory(
          rag_chain,
          self.get_session_history,
          input_messages_key="input",
          history_messages_key="chat_history",
          output_messages_key="answer",
      )
      
      response = conversational_rag_chain.invoke(
        {"input": text},
        config={"configurable": {"session_id": "abc123"}},  # constructs a key "abc123" in `store`.
      )

      # print(response['answer'])
      return (response, self.store["abc123"])

    
    
    # CRUD Section
    def get_document(self, uuid: str): 
        self.db.get_by_ids([uuid])
      
    def add_documents(self, document_list: list) -> None:
        uuids = [str(uuid4()) for _ in range(len(document_list))]
        self.db.add_documents(documents=document_list, ids=uuids)

    def update_document(self, doc_id, new_document):
        self.db.update_document(document_id=doc_id, document=new_document)
    
    
    ## Load Docs Section
    def loadDoc(self, fileName:str):
      loader = PyPDFLoader(f"./assets/docs/{fileName}")
      docs = loader.load()
      splits = self.text_splitter.split_documents(docs)

      # Save each doc on a collection
      self.add_documents(splits)
      print(f"Saved {fileName}")
      
      
      
    def loadAllDocs(self):
      onlyfiles = [f for f in listdir("./assets/docs") if isfile(join("./assets/docs", f))]
      print(f"To Load files: {onlyfiles}")
      for fileName in onlyfiles:
        self.loadDoc(fileName)
      
      print("Loaded all files")      