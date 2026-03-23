"""
Universal RAG Test Script
Works with BOTH Anthropic and Ollama!
Just change the config and it switches automatically
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def main():
    print("=" * 80)
    print("UNIVERSAL RAG CHATBOT")
    print("=" * 80)
    
    # Choose provider
    print("\n🤖 Choose LLM Provider:")
    print("  1. Ollama (FREE, Local)")
    print("  2. Anthropic Claude (Paid, Cloud)")
    
    choice = input("\nChoice (1/2): ").strip()
    
    if choice == '1':
        provider = 'ollama'
        print("\n✅ Selected: Ollama (FREE)")
        print("💡 Make sure Ollama is running: ollama serve")
        
        # Check Ollama
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                print("✅ Ollama is running")
            else:
                print("❌ Ollama not responding")
                return
        except:
            print("❌ Cannot connect to Ollama")
            print("💡 Start with: ollama serve")
            return
        
        # Import Ollama version
        from rag_pipeline.ollama_chatbot import OllamaRAGChatbot
        
        # Choose model
        print("\n📦 Available Models:")
        print("  1. llama3      (Recommended)")
        print("  2. mistral     (Fast)")
        print("  3. phi3        (Small)")
        print("  4. Custom")
        
        model_choice = input("\nModel (1-4): ").strip()
        models = {'1': 'llama3', '2': 'mistral', '3': 'phi3'}
        
        if model_choice == '4':
            model = input("Enter model name: ").strip()
        else:
            model = models.get(model_choice, 'llama3')
        
        print(f"\n🤖 Initializing with {model}...")
        
        try:
            chatbot = OllamaRAGChatbot(
                model=model,
                storage_dir='rag_storage_ollama'
            )
        except Exception as e:
            print(f"❌ Error: {e}")
            return
    
    elif choice == '2':
        provider = 'anthropic'
        print("\n✅ Selected: Anthropic Claude")
        
        # Get API key
        try:
            from config import ANTHROPIC_API_KEY
        except:
            ANTHROPIC_API_KEY = None
        
        if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY == '':
            ANTHROPIC_API_KEY = input("\nEnter Anthropic API key: ").strip()
        
        if not ANTHROPIC_API_KEY:
            print("❌ No API key provided")
            return
        
        # Import Anthropic version
        from rag_pipeline.standalone_chatbot import StandaloneRAGChatbot
        
        print("\n🤖 Initializing with Claude Sonnet 4...")
        
        try:
            chatbot = StandaloneRAGChatbot(
                anthropic_api_key=ANTHROPIC_API_KEY,
                storage_dir='rag_storage_anthropic'
            )
        except Exception as e:
            print(f"❌ Error: {e}")
            return
    
    else:
        print("❌ Invalid choice")
        return
    
    print(f"\n✅ {provider.upper()} RAG Chatbot initialized!")
    print(f"📚 Loaded documents: {len(chatbot.list_documents())}")
    
    # Main loop
    while True:
        print("\n" + "=" * 80)
        print("OPTIONS:")
        print("  1. Add PDF document")
        print("  2. Add text document")
        print("  3. List documents")
        print("  4. Chat with documents")
        print("  5. Remove document")
        print("  6. Clear all")
        print("  7. Switch provider")
        print("  8. Exit")
        print("=" * 80)
        
        option = input("\nChoice: ").strip()
        
        if option == '1':
            # Add PDF
            pdf_path = input("PDF path: ").strip()
            
            if not os.path.exists(pdf_path):
                print("❌ File not found!")
                continue
            
            doc_id = input("Document ID (Enter for filename): ").strip()
            if not doc_id:
                doc_id = os.path.basename(pdf_path).replace('.pdf', '')
            
            print(f"\n📄 Processing...")
            success = chatbot.add_document_from_file(pdf_path, doc_id)
            
            if success:
                print(f"✅ Added: {doc_id}")
            else:
                print("❌ Failed")
        
        elif option == '2':
            # Add text
            doc_id = input("Document ID: ").strip()
            print("Text (type 'END' when done):")
            
            lines = []
            while True:
                line = input()
                if line == 'END':
                    break
                lines.append(line)
            
            text = '\n'.join(lines)
            chatbot.add_document_from_text(text, doc_id)
            print(f"✅ Added: {doc_id}")
        
        elif option == '3':
            # List documents
            docs = chatbot.list_documents()
            
            if not docs:
                print("\n📭 No documents")
            else:
                print(f"\n📚 Documents ({len(docs)}):")
                for doc in docs:
                    print(f"  • {doc['doc_id']} ({doc['num_chunks']} chunks)")
        
        elif option == '4':
            # Chat
            if not chatbot.list_documents():
                print("\n❌ No documents! Add some first.")
                continue
            
            print("\n💬 Chat Mode")
            print("Commands: 'back' to exit, 'clear' to clear history")
            print("=" * 80)
            
            while True:
                query = input("\n🧑 You: ").strip()
                
                if query.lower() == 'back':
                    break
                
                if query.lower() == 'clear':
                    chatbot.clear_history()
                    print("✅ History cleared")
                    continue
                
                if not query:
                    continue
                
                print(f"\n🤖 {provider.capitalize()} is thinking...")
                
                response = chatbot.chat(
                    query,
                    top_k=5,
                    use_reranking=True,
                    temperature=0.7
                )
                
                print(f"\n🤖 Assistant:\n{response}\n")
        
        elif option == '5':
            # Remove
            docs = chatbot.list_documents()
            
            if not docs:
                print("\n📭 No documents")
                continue
            
            print("\n📚 Documents:")
            for i, doc in enumerate(docs, 1):
                print(f"  {i}. {doc['doc_id']}")
            
            doc_id = input("\nDocument ID to remove: ").strip()
            chatbot.remove_document(doc_id)
            print(f"✅ Removed: {doc_id}")
        
        elif option == '6':
            # Clear all
            confirm = input("\n⚠️  Clear ALL? (yes/no): ").strip().lower()
            if confirm == 'yes':
                chatbot.document_store.clear_all()
                chatbot.clear_history()
                print("✅ Cleared")
        
        elif option == '7':
            # Switch provider
            print("\n🔄 Restart script to switch providers")
            print("💡 Or change LLM_PROVIDER in rag_pipeline/config_rag.py")
        
        elif option == '8':
            # Exit
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()