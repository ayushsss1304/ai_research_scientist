import sys
import os

with open("dummy_test.txt", "w") as f:
    f.write("Welcome to the research api!")

sys.path.insert(0, os.getcwd())
try:
    from rag_pipeline.ollama_chatbot import OllamaRAGChatbot
    bot = OllamaRAGChatbot()
    print("Bot initialized")
    res = bot.add_document_from_file("dummy_test.txt", "dummy_test")
    print("Add file result:", res)
except Exception as e:
    import traceback
    traceback.print_exc()
