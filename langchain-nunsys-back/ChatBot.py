import json
import asyncio
import collections
import os

from vectorstoreController import VectorstoreController
from typing import List
from faiss import IndexFlatL2
from LlmController import LangchainLlms
from ChatOpenAI import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_openai import OpenAIEmbeddings
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    Document,
    BaseMessage
)

class ChatBot:
  def __init__(self):
    # Initial Prompt
    systemPrompt = f"Eres el asistente para entrevistas de Ricardo, que quiere ser contratado con Ingeniero de LLMs"
    self.__systemPrompt = SystemMessage(content=systemPrompt)

    # Embeddings
    # self.__embeddings = OpenAIEmbeddings()
    
    # VectorStore
    self.__memory = FAISS(
      embedding_function= OpenAIEmbeddings(),
      index=IndexFlatL2,
      docstore=InMemoryDocstore({}),
      index_to_docstore_id={}
    )
    
    self.__chat_history_buffer = collections.deque([], maxlen=5)
    self.__llm = ChatOpenAI()
    self.__messages = [self.__systemPrompt]

  def getLlm(self):
    return self.__llm
  
  # Combinar docs y el chat
  def combineDocsAndChatHistory(self, relevantMessages: list) -> List[BaseMessage]:
    if not relevantMessages:
      return []

    nonDuplicateMessages = []
    
    # Add only new info
    for message in relevantMessages:
      if message not in self.__chat_history_buffer:
        nonDuplicateMessages.append(message)
    messages = []
    messages.append(self.__systemPrompt)

    for message in nonDuplicateMessages:
      msgDoc = json.loads(message)
      messages.append(HumanMessage(content=msgDoc["query"]))
      messages.append(AIMessage(content=msgDoc["answer"]))
    for message in self.__chat_history_buffer:
      msgDoc = json.loads(message) if isinstance(message, str) else message
      messages.append(HumanMessage(content=msgDoc["query"]))
      messages.append(AIMessage(content=msgDoc["answer"]))

    return messages
  
  def addToMemory(self, data: str):
    self.__memory.add_documents([Document(page_content=data)])
    print(f"added {data} to memory")
  
  # Metodo principal de chat
  def chat(self, *, userQuery: str, cancelToken=None):
    # Use RAG
    (results, history) = vecController.query(userQuery)
    
    print(f"Respuesta: {results["answer"]}")
    print("History:")
    for message in history.messages:
      if isinstance(message, AIMessage):
          prefix = "AI"
      else:
          prefix = "User"

      print(f"{prefix}: {message.content}\n")


  async def test_chatbot(self):
    while True:
      query = input("enter your query: ")
      if query.lower() == "exit":
        return

      self.chat(userQuery=query)
      # async for token, _ in self.chat(userQuery=query):
      #   print(token, end='')

      print("\n")
      print("*" * 100)
      print("\n")

if __name__ == '__main__':
    os.environ["OPENAI_API_KEY"] = ""
    
    bot = ChatBot()
    vecController = VectorstoreController(bot.getLlm())
    # vecController.loadAllDocs()
    
    loop = asyncio.get_event_loop()
    loop.run_until_complete(bot.test_chatbot())
    
    # Create an instance of the controller
    
    # vecController.add_documents([document_1])
    # vecController.update_document("6005f2c8-9a0d-4eb4-a81f-f973f48da61d", document_new)
    
