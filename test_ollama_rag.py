"""
Test Ollama RAG Chatbot
Completely FREE and runs locally!
"""

from rag_pipeline.ollama_chatbot import OllamaRAGChatbot
import os

def main():
    print("=" * 80)
    print("OLLAMA RAG CHATBOT TEST (FREE & LOCAL!)")
    print("=" * 80)
    
    # Choose model
    print("\n📦 Available Ollama Models:")
    print("  1. llama3      (Recommended, 8B params)")
    print("  2. llama3:70b  (Best quality, needs 40GB RAM)")
    print("  3. mistral     (Fast, 7B params)")
    print("  4. phi3        (Small, 3.8B params)")
    print("  5. gemma2      (Google, 9B params)")
    print("  6. Custom")
    
    choice = input("\nSelect model (1-6) or press Enter for llama3: ").strip()
    
    model_map = {
        '1': 'llama3',
        '2': 'llama3:70b',
        '3': 'mistral',
        '4': 'phi3',
        '5': 'gemma2',
        '': 'llama3'
    }
    
    if choice == '6':
        model = input("Enter custom model name: ").strip()
    else:
        model = model_map.get(choice, 'llama3')
    
    print(f"\n✅ Selected model: {model}")
    print("\n💡 If model not found, run: ollama pull", model)
    
    # Initialize chatbot
    print("\n🤖 Initializing Ollama RAG Chatbot...")
    try:
        chatbot = OllamaRAGChatbot(
            model=model,
            ollama_url="http://localhost:11434",
            storage_dir='rag_storage_ollama'  # Separate from Anthropic
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Make sure Ollama is running:")
        print("   1. Install: https://ollama.ai/download")
        print("   2. Run: ollama serve")
        print(f"   3. Pull model: ollama pull {model}")
        return
    
    print("\n✅ Ollama RAG Chatbot initialized!")
    print(f"📁 Storage location: rag_storage_ollama/")
    print(f"📚 Loaded documents: {len(chatbot.list_documents())}")
    
    # Show menu
    while True:
        print("\n" + "=" * 80)
        print("OPTIONS:")
        print("  1. Add PDF document")
        print("  2. Add text document")
        print("  3. List documents")
        print("  4. Chat with documents")
        print("  5. Remove document")
        print("  6. Clear all documents")
        print("  7. Change model")
        print("  8. Exit")
        print("=" * 80)
        
        choice = input("\nChoice: ").strip()
        
        if choice == '1':
            # Add PDF
            pdf_path = input("Enter PDF path: ").strip()
            if not os.path.exists(pdf_path):
                print("❌ File not found!")
                continue
            
            doc_id = input("Enter document ID (or press Enter for filename): ").strip()
            if not doc_id:
                doc_id = os.path.basename(pdf_path).replace('.pdf', '')
            
            print(f"\n📄 Processing {pdf_path}...")
            success = chatbot.add_document_from_file(pdf_path, doc_id)
            
            if success:
                print(f"✅ Document added: {doc_id}")
            else:
                print("❌ Failed to add document")
        
        elif choice == '2':
            # Add text
            doc_id = input("Enter document ID: ").strip()
            print("Enter text (type 'END' on a new line when done):")
            
            lines = []
            while True:
                line = input()
                if line == 'END':
                    break
                lines.append(line)
            
            text = '\n'.join(lines)
            
            if text:
                chatbot.add_document_from_text(text, doc_id)
                print(f"✅ Text document added: {doc_id}")
        
        elif choice == '3':
            # List documents
            docs = chatbot.list_documents()
            
            if not docs:
                print("\n📭 No documents loaded")
            else:
                print(f"\n📚 Loaded Documents ({len(docs)}):")
                print("-" * 80)
                for doc in docs:
                    print(f"\n  ID: {doc['doc_id']}")
                    print(f"  Chunks: {doc['num_chunks']}")
                    print(f"  Added: {doc['added_at']}")
                    if doc['metadata']:
                        print(f"  Metadata: {doc['metadata']}")
        
        elif choice == '4':
            # Chat
            if not chatbot.list_documents():
                print("\n❌ No documents loaded! Add some documents first.")
                continue
            
            print("\n💬 Chat Mode (type 'back' to return, 'clear' to clear history)")
            print("=" * 80)
            
            while True:
                query = input("\nYou: ").strip()
                
                if query.lower() == 'back':
                    break
                
                if query.lower() == 'clear':
                    chatbot.clear_history()
                    print("✅ History cleared")
                    continue
                
                if not query:
                    continue
                
                print("\n🤔 Thinking... (Ollama is generating locally)")
                
                # Chat with custom temperature
                temp = 0.7  # Default
                response = chatbot.chat(
                    query, 
                    top_k=10, 
                    use_reranking=True,
                    temperature=temp
                )
                
                print(f"\n🤖 Assistant: {response}\n")
        
        elif choice == '5':
            # Remove document
            docs = chatbot.list_documents()
            
            if not docs:
                print("\n📭 No documents to remove")
                continue
            
            print("\n📚 Documents:")
            for i, doc in enumerate(docs, 1):
                print(f"  {i}. {doc['doc_id']}")
            
            doc_id = input("\nEnter document ID to remove: ").strip()
            chatbot.remove_document(doc_id)
            print(f"✅ Removed: {doc_id}")
        
        elif choice == '6':
            # Clear all
            confirm = input("\n⚠️  Clear ALL documents? (yes/no): ").strip().lower()
            if confirm == 'yes':
                chatbot.document_store.clear_all()
                chatbot.clear_history()
                print("✅ All documents and history cleared")
        
        elif choice == '7':
            # Change model
            new_model = input("\nEnter new model name: ").strip()
            if new_model:
                print(f"\n🔄 Switching to {new_model}...")
                try:
                    chatbot = OllamaRAGChatbot(
                        model=new_model,
                        storage_dir='rag_storage_ollama'
                    )
                    print(f"✅ Switched to {new_model}")
                except Exception as e:
                    print(f"❌ Error: {e}")
        
        elif choice == '8':
            # Exit
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice")

if __name__ == '__main__':
    main()