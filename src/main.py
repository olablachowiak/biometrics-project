from src.chat import LlamaChat
import unittest

def main():
  print("Llama LLM Chat - Type 'exit' to quit.")

  chat = LlamaChat()
  messages = []

  while True:
    user_input = input("\nYou: ")

    if user_input.lower() in ["exit", "quit"]:
      print("\nExiting. Goodbye!")
      break

    messages.append({"role": "user", "content": user_input})
    response = chat.get_response(messages)
    messages.append({"role": "assistant", "content": response})
    print(f"Llama: {response}")

if __name__ == "__main__":
  main()