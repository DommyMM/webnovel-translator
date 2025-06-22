import json
import os
from pathlib import Path
from typing import Dict, List
import chromadb
from chromadb.utils import embedding_functions
from chromadb.utils.embedding_functions import EmbeddingFunction
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

class Qwen3EmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name="Qwen/Qwen3-Embedding-8B"):
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading embedding model: {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)
        print(f"Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def __call__(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

class RAGDebugger:
    def __init__(self, use_qwen3=True):
        self.use_qwen3 = use_qwen3
        self.db_path = "../data/terminology/chroma_db_rag"
        self.collection_name = "rag_terminology"
        self.chapter_file = "../data/chapters/chinese/chapter_0001_cn.txt"
        
        print("=" * 60)
        print("RAG DEBUG TOOL")
        print("=" * 60)
        
        self.load_database()
        self.load_chapter()
    
    def load_database(self):
        print(f"\n1. LOADING DATABASE")
        print(f"   Database path: {self.db_path}")
        print(f"   Collection: {self.collection_name}")
        
        if not Path(self.db_path).exists():
            print(f"   ERROR: Database not found at {self.db_path}")
            return
        
        # Create client
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        # Setup embedding function
        if self.use_qwen3:
            try:
                self.embedding_function = Qwen3EmbeddingFunction("Qwen/Qwen3-Embedding-8B")
                print(f"   Using Qwen3 embeddings")
            except Exception as e:
                print(f"   Qwen3 failed: {e}")
                print(f"   Falling back to basic embeddings")
                self.use_qwen3 = False
                self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="paraphrase-multilingual-MiniLM-L12-v2"
                )
        else:
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="paraphrase-multilingual-MiniLM-L12-v2"
            )
            print(f"   Using basic sentence-transformers")
        
        # Get collection
        try:
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            count = self.collection.count()
            print(f"   SUCCESS: Collection loaded with {count} terms")
        except Exception as e:
            print(f"   ERROR: Failed to load collection: {e}")
            print(f"   Available collections: {self.client.list_collections()}")
            return
    
    def load_chapter(self):
        print(f"\n2. LOADING CHAPTER")
        print(f"   Chapter file: {self.chapter_file}")
        
        if not Path(self.chapter_file).exists():
            print(f"   ERROR: Chapter file not found")
            return
        
        with open(self.chapter_file, 'r', encoding='utf-8') as f:
            self.chapter_text = f.read().strip()
        
        print(f"   SUCCESS: Loaded {len(self.chapter_text)} characters")
        print(f"   Preview: {self.chapter_text[:100]}...")
    
    def show_database_contents(self):
        print(f"\n3. DATABASE CONTENTS")
        print("-" * 40)
        
        try:
            # Get all documents
            all_docs = self.collection.get(limit=50)
            
            print(f"Total terms in database: {len(all_docs['documents'])}")
            print("\nFirst 10 terms:")
            for i, (doc, metadata) in enumerate(zip(all_docs['documents'], all_docs['metadatas'])):
                if i >= 10:
                    break
                english = metadata.get('english_term', 'N/A')
                category = metadata.get('category', 'N/A')
                print(f"   {i+1:2d}. {doc} → {english} ({category})")
                
        except Exception as e:
            print(f"   ERROR: Could not retrieve database contents: {e}")
    
    def test_similarity_thresholds(self):
        print(f"\n4. TESTING SIMILARITY THRESHOLDS")
        print("-" * 40)
        
        if not hasattr(self, 'chapter_text'):
            print("   ERROR: No chapter text loaded")
            return
        
        # Test different thresholds
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        for threshold in thresholds:
            print(f"\n   Threshold {threshold}:")
            try:
                results = self.collection.query(
                    query_texts=[self.chapter_text],
                    n_results=10,
                    include=['documents', 'metadatas', 'distances']
                )
                
                matches = 0
                for doc, metadata, distance in zip(
                    results['documents'][0], 
                    results['metadatas'][0], 
                    results['distances'][0]
                ):
                    similarity = 1.0 - distance
                    if similarity >= threshold:
                        matches += 1
                        if matches <= 3:  # Show first 3 matches
                            english = metadata.get('english_term', 'N/A')
                            print(f"     {doc} → {english} (sim: {similarity:.3f})")
                
                print(f"     Total matches: {matches}")
                
            except Exception as e:
                print(f"     ERROR: {e}")
    
    def test_specific_terms(self):
        print(f"\n5. TESTING SPECIFIC TERMS")
        print("-" * 40)
        
        # Test specific Chinese terms that should be in your chapter
        test_terms = [
            "龙尘",      # Long Chen (main character)
            "丹帝",      # Pill God  
            "九星霸体诀", # Nine Star Hegemon Body Art
            "聚气境",    # Qi Condensation realm
            "筑基期",    # Foundation Building stage
            "修炼",      # Cultivation
            "丹田"       # Dantian
        ]
        
        for term in test_terms:
            print(f"\n   Testing: {term}")
            
            # Check if term exists in chapter
            in_chapter = term in self.chapter_text
            print(f"     In chapter: {in_chapter}")
            
            if in_chapter:
                # Query database for this specific term
                try:
                    results = self.collection.query(
                        query_texts=[term],
                        n_results=3,
                        include=['documents', 'metadatas', 'distances']
                    )
                    
                    print(f"     Database matches:")
                    for doc, metadata, distance in zip(
                        results['documents'][0], 
                        results['metadatas'][0], 
                        results['distances'][0]
                    ):
                        similarity = 1.0 - distance
                        english = metadata.get('english_term', 'N/A')
                        print(f"       {doc} → {english} (sim: {similarity:.3f})")
                        
                except Exception as e:
                    print(f"     ERROR: {e}")
    
    def run_full_debug(self):
        if hasattr(self, 'collection') and hasattr(self, 'chapter_text'):
            self.show_database_contents()
            self.test_similarity_thresholds()
            self.test_specific_terms()
        else:
            print("ERROR: Database or chapter not loaded properly")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug RAG System")
    parser.add_argument("--no-qwen", action="store_true", help="Use basic embeddings")
    
    args = parser.parse_args()
    
    use_qwen3 = not args.no_qwen
    
    debugger = RAGDebugger(use_qwen3=use_qwen3)
    debugger.run_full_debug()

if __name__ == "__main__":
    main()