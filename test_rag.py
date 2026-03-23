"""
Test Standalone RAG Chatbot
This is COMPLETELY INDEPENDENT from scraper and knowledge graph
"""

from rag_pipeline.standalone_chatbot import StandaloneRAGChatbot
import os

def main():
    print("=" * 80)
    print("STANDALONE RAG CHATBOT TEST")
    print("=" * 80)
    
    # Get API key from config
    try:
        from config import ANTHROPIC_API_KEY
    except:
        ANTHROPIC_API_KEY = input("\nEnter your Anthropic API key: ").strip()
    
    if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY == '':
        print("\n❌ No Anthropic API key provided!")
        print("Get one at: https://console.anthropic.com/")
        return
    
    # Initialize chatbot (independent system)
    print("\n🤖 Initializing RAG Chatbot...")
    chatbot = StandaloneRAGChatbot(
        anthropic_api_key=ANTHROPIC_API_KEY,
        storage_dir='rag_storage'  # Separate storage
    )
    
    print("\n✅ RAG Chatbot initialized!")
    print(f"📁 Storage location: rag_storage/")
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
        print("  7. Exit")
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
            
            print("\n💬 Chat Mode (type 'back' to return to menu)")
            print("-" * 80)
            
            while True:
                query = input("\nYou: ").strip()
                
                if query.lower() == 'back':
                    break
                
                if not query:
                    continue
                
                print("\n🤔 Thinking...")
                response = chatbot.chat(query, top_k=5, use_reranking=True)
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
            # Exit
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice")

if __name__ == '__main__':
    main()