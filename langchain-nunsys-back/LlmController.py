import asyncio

# from OpenAI import OpenAI
from typing import Union, List
# from AzureOpenAI import AzureOpenAI
from pydantic import BaseModel, Extra
from langchain.schema.messages import BaseMessage
from langchain.chat_models.base import BaseChatModel
from langchain.base_language import BaseLanguageModel
from callbacks.callbacks import AsyncQueueCallbackHandler

class Llm(BaseModel):
  llm: BaseChatModel
  llmName: str
  llmArgs: dict
  callback: list = None
  
  """Pydantic conf"""
  class Config: 
    extra = Extra.allow
    arbitrary_types_allowed = True
  
  def __str__(self):
    return f"LLM: {self.name} \n Args: {self.llmArgs}"
  
class LangchainLlms:
  def __init__(self):
    self.__llms= {
      "OpenAI": {
        "llm": OpenAI,
        "schema": OpenAI
      }
    }
  # def getLlm(self, llmName: str, callback=None, **llm_kwargs) -> Llm: 
  #   print(f"getLlm {llmName}")
  #   if(llmName not in self.__llms): 
  #     raise ValueError(f"Invalid LLM name, must be one of {self.__llms.keys()}") 
    
  #   llm = self.__llms[llmName]["llm"]
  #   llmArgs = self.__llms[llmName]["schema"](**llm_kwargs)
  #   llmObj = llm(**dict(llmArgs))
    
  #   return Llm(llm=llmObj, llmArgs=dict(llmArgs), llmName=llmName, callback=callback)
  
  # @staticmethod
  def trySetStreamingOptions(llm: Union[BaseChatModel, BaseLanguageModel]) -> Union[
      BaseChatModel, BaseLanguageModel]:
      # If the LLM type is OpenAI or ChatOpenAI, set streaming to True
      if isinstance(llm, BaseLanguageModel) or isinstance(llm, BaseChatModel):
          if hasattr(llm, "streaming") and isinstance(llm.streaming, bool):
              llm.streaming = True
              print("set streaming to true")
          elif hasattr(llm, "stream") and isinstance(llm.stream, bool):
              print("set streaming to true")
              llm.stream = True

      return llm
  
  @staticmethod
  def getNonStreamResponse(llm: BaseChatModel, messages: List[BaseMessage]) -> str: 
    res = llm.predict_messages(messages)
    print("getNonStreamResponse res: ")
    print(res)
    return res.content

  # @staticmethod
  async def getStreamResponse(llm: BaseChatModel, messages: List[BaseMessage], cancelToken=None):
      llm = LangchainLlms.trySetStreamingOptions(llm)
      queue = asyncio.Queue()
      callback = AsyncQueueCallbackHandler(queue=queue, cancelToken=cancelToken)

      if hasattr(llm, "callbacks"):
          llm.callbacks = [callback]

      task = asyncio.create_task(llm.agenerate(messages=[messages]))
      token = ""

      while True:
        if cancelToken and cancelToken.isCancelled is True:
          break

        try:
          token = await asyncio.wait_for(queue.get(), timeout=60 * 3)

          if token == "<END OF LLM RESPONSE>":
              break

        except asyncio.TimeoutError:
          print("Consumer timed out waiting for a token.")
          task.cancel()
          raise asyncio.TimeoutError("Consumer timed out waiting for a token.")

        yield token
