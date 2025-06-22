from ollama import chat
from .config import (
  OLLAMA_MODEL
)

class LlamaChat:
  def get_response(self, messages):
    response = chat(
        model=OLLAMA_MODEL,
        messages=messages,
    )
    return response.message.content