# from pydantic import validator, Extra
from langchain_openai import ChatOpenAI
# https://python.langchain.com/v0.2/api_reference/openai/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html

# class OpenAI(ChatOpenAI):

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.5,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # api_key="...",  # if you prefer to pass api key in directly instaed of using env vars
    # base_url="...",
    # organization="...",
    # other params...
)

# class OpenAI(ChatOpenAI):
#     model_name: str = "gpt-3.5-turbo"
#     temperature: float = 0
#     openai_api_key: str
#     streaming: bool = True

    # class Config:
    #     """Configuration for this pydantic object."""

    #     extra = Extra.allow
    #     arbitrary_types_allowed = True

    # @staticmethod
    # def getValidModelNames():
    #     validModelNames = {"gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-3.5-turbo-0613"}
    #     return validModelNames
      
    # ## Validaciones  
    # @validator("temperature")
    # def validateTemperature(cls, request):
    #     if request < 0 or request > 1:
    #         raise ValueError("Temperature must be between 0 and 1")

    #     return request

    # @validator("model_name")
    # def validateModelName(cls, request):
    #     validModelNames = cls.getValidModelNames()

    #     if request not in validModelNames:
    #         raise ValueError(f"invalid model name given - {request} , valid ones are {validModelNames}")
    #     return request