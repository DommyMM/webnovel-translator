import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import chromadb
from chromadb.utils import embedding_functions
from chromadb.utils.embedding_functions import EmbeddingFunction
from sentence_transformers import SentenceTransformer

def chunk_chinese_text_by_lines(text):
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if len(line) >= 5 and not line.startswith('第') and not line.startswith('﻿'):
            cleaned_lines.append(line)
    return cleaned_lines

def chunk_chinese_text_by_lines_and_sentences(text):
    """Split text by lines first, then by punctuation for better semantic chunking"""
    lines = text.split('\n')
    all_chunks = []
    
    # Chinese and English punctuation that marks sentence boundaries
    sentence_endings = ['？', '。', '！', '；', '?', '.', '!', ';']
    
    for line in lines:
        line = line.strip()
        if len(line) < 5 or line.startswith('第') or line.startswith('﻿'):
            continue
            
        # Split line by sentence-ending punctuation
        current_sentence = ""
        for char in line:
            current_sentence += char
            
            # If we hit sentence-ending punctuation, finalize the sentence
            if char in sentence_endings:
                # Clean the sentence
                sentence = current_sentence.strip()
                # Remove quotes and extra punctuation from ends
                sentence = sentence.strip('"').strip('"').strip('——').strip()
                
                if len(sentence) >= 5:  # Only keep substantial sentences
                    all_chunks.append(sentence)
                
                current_sentence = ""
        
        # Add any remaining text as a sentence
        if current_sentence.strip():
            sentence = current_sentence.strip()
            sentence = sentence.strip('"').strip('"').strip('——').strip()
            if len(sentence) >= 5:
                all_chunks.append(sentence)
    
    return all_chunks

class BGEEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name="BAAI/bge-m3"):
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        print(f"Loaded {model_name} on {device}")
    
    def __call__(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

class Qwen3EmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name="Qwen/Qwen3-Embedding-8B"):
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        print(f"Loaded {model_name} on {device}")
    
    def __call__(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

class RAGRetrievalTester:
    def __init__(self, use_qwen3=True, use_bge=False):
        self.use_qwen3 = use_qwen3
        self.use_bge = use_bge
        
        if use_bge:
            self.db_path = "../data/terminology/chroma_db_bge"
            self.collection_name = "bge_terminology"
            self.model_name = "BGE-M3"
        elif use_qwen3:
            self.db_path = "../data/terminology/chroma_db_rag"
            self.collection_name = "rag_terminology"
            self.model_name = "Qwen3-8B"
        else:
            self.db_path = "../data/terminology/chroma_db"
            self.collection_name = "basic_terminology" 
            self.model_name = "MPNet"

        self.load_database()
    
    def load_database(self):
        if not Path(self.db_path).exists():
            raise FileNotFoundError(f"ChromaDB database not found: {self.db_path}")
        
        print(f"Loading ChromaDB from: {self.db_path}")
        print(f"Using embedding model: {self.model_name}")
        
        # Create client
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        # Setup embedding function
        if self.use_bge:
            self.embedding_function = BGEEmbeddingFunction("BAAI/bge-m3")
        elif self.use_qwen3:
            self.embedding_function = Qwen3EmbeddingFunction("Qwen/Qwen3-Embedding-8B")
        else:
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
            )
        
        # Get collection
        self.collection = self.client.get_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function
        )
        
        count = self.collection.count()
        print(f"Collection loaded: {self.collection_name} ({count} terms)")
    
    def test_specific_queries(self):
        """Test specific query patterns that should match known terms"""
        
        test_cases = [
            {
                "query": "绝世丹帝——龙尘",
                "expected": ["丹帝", "龙尘"],
                "description": "Should find Pill God and Long Chen"
            },
            {
                "query": "虎骨丹",
                "expected": ["虎骨丹"],
                "description": "Should find Tiger Bone Pill"
            },
            {
                "query": "药师大人",
                "expected": ["药师"],
                "description": "Should find Master Alchemist"
            },
            {
                "query": "龙夫人",
                "expected": ["龙夫人"],
                "description": "Should find Mrs. Long"
            },
            {
                "query": "灵根被吸走了",
                "expected": ["灵根"],
                "description": "Should find Spiritual Root"
            }
        ]
        
        print(f"\nTesting specific queries ({self.model_name})")
        print("=" * 60)
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\nTest {i}: {test_case['description']}")
            print(f"Query: '{test_case['query']}'")
            print(f"Expected: {test_case['expected']}")
            
            # Query ChromaDB
            results = self.collection.query(
                query_texts=[test_case['query']],
                n_results=10,
                include=['documents', 'metadatas', 'distances']
            )
            
            print(f"Results found: {len(results['documents'][0])}")
            
            found_terms = []
            for doc, metadata, distance in zip(
                results['documents'][0], 
                results['metadatas'][0], 
                results['distances'][0]
            ):
                chinese_term = doc
                english_term = metadata['english_term']
                similarity = 1.0 - distance
                
                print(f"  {chinese_term} → {english_term} (similarity: {similarity:.3f})")
                found_terms.append(chinese_term)
            
            # Check if expected terms were found
            missing = set(test_case['expected']) - set(found_terms)
            if missing:
                print(f"Missing: {list(missing)}")
            else:
                print(f"All expected terms found")
    
    def test_threshold_sensitivity(self):
        """Test how similarity threshold affects retrieval"""
        
        query = "绝世丹帝——龙尘"
        thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        
        print(f"\nThreshold sensitivity test ({self.model_name})")
        print("=" * 60)
        print(f"Query: '{query}'")
        
        # Get raw results
        results = self.collection.query(
            query_texts=[query],
            n_results=20,
            include=['documents', 'metadatas', 'distances']
        )
        
        print(f"\nRaw results (top 20):")
        all_results = []
        for doc, metadata, distance in zip(
            results['documents'][0], 
            results['metadatas'][0], 
            results['distances'][0]
        ):
            chinese_term = doc
            english_term = metadata['english_term']
            similarity = 1.0 - distance
            all_results.append((chinese_term, english_term, similarity))
            print(f"  {chinese_term} → {english_term} (sim: {similarity:.3f})")
        
        print(f"\nFiltering by thresholds:")
        for threshold in thresholds:
            filtered = [(ch, en, sim) for ch, en, sim in all_results if sim >= threshold]
            print(f"  Threshold {threshold}: {len(filtered)} terms")
            for ch, en, sim in filtered[:5]:  # Show top 5
                print(f"    • {ch} → {en} ({sim:.3f})")
    
    def test_all_database_terms(self):
        """Show all terms in the database for reference"""
        
        print(f"\nAll database terms")
        print("=" * 60)
        
        # Get all documents
        all_docs = self.collection.get(include=['documents', 'metadatas'])
        
        terms_by_category = {}
        for doc, metadata in zip(all_docs['documents'], all_docs['metadatas']):
            chinese_term = doc
            english_term = metadata['english_term']
            category = metadata.get('category', 'unknown')
            
            if category not in terms_by_category:
                terms_by_category[category] = []
            terms_by_category[category].append((chinese_term, english_term))
        
        for category, terms in terms_by_category.items():
            print(f"\n{category.upper()} ({len(terms)} terms):")
            for chinese_term, english_term in sorted(terms):
                print(f"  • {chinese_term} → {english_term}")
    
    def test_line_by_line_chapter1(self):
        """Test line-by-line retrieval on Chapter 1"""
        
        print(f"\nLine-by-line Chapter 1 test ({self.model_name})")
        print("=" * 60)
        
        # Load Chapter 1
        chapter_file = "../data/chapters/clean/chapter_0001_cn.txt"
        if not Path(chapter_file).exists():
            print(f"Chapter file not found: {chapter_file}")
            return
        
        with open(chapter_file, 'r', encoding='utf-8') as f:
            chinese_text = f.read().strip()
        
        lines = chunk_chinese_text_by_lines(chinese_text)
        print(f"Split into {len(lines)} lines")
        
        total_found = {}
        
        for i, line in enumerate(lines[:10], 1):  # Test first 10 lines
            print(f"\n📝 Line {i}: {line[:50]}...")
            
            results = self.collection.query(
                query_texts=[line],
                n_results=10,
                include=['documents', 'metadatas', 'distances']
            )
            
            line_terms = []
            for doc, metadata, distance in zip(
                results['documents'][0], 
                results['metadatas'][0], 
                results['distances'][0]
            ):
                chinese_term = doc
                english_term = metadata['english_term']
                similarity = 1.0 - distance
                
                if similarity >= 0.1:  # Apply threshold
                    line_terms.append((chinese_term, english_term, similarity))
                    if chinese_term not in total_found:
                        total_found[chinese_term] = (english_term, similarity)
                    elif similarity > total_found[chinese_term][1]:
                        total_found[chinese_term] = (english_term, similarity)
            
            if line_terms:
                print(f"  Found {len(line_terms)} terms:")
                for chinese_term, english_term, similarity in line_terms:
                    print(f"    {chinese_term} → {english_term} ({similarity:.3f})")
            else:
                print(f"  No terms found above threshold")
        
        print(f"\nTotal unique terms found:")
        for chinese_term, (english_term, similarity) in sorted(total_found.items()):
            print(f"  {chinese_term} → {english_term} ({similarity:.3f})")
    
    def test_line_by_line_chapter1(self):
        """Test line-by-line + sentence retrieval on Chapter 1"""
        
        print(f"\nLine-by-line + sentence Chapter 1 test ({self.model_name})")
        print("=" * 60)
        
        # Load Chapter 1
        chapter_file = "../data/chapters/clean/chapter_0001_cn.txt"
        if not Path(chapter_file).exists():
            print(f"Chapter file not found: {chapter_file}")
            return
        
        with open(chapter_file, 'r', encoding='utf-8') as f:
            chinese_text = f.read().strip()
        
        # Test both old and new chunking
        old_lines = chunk_chinese_text_by_lines(chinese_text)
        new_chunks = chunk_chinese_text_by_lines_and_sentences(chinese_text)
        
        print(f"Old chunking: {len(old_lines)} lines")
        print(f"New chunking: {len(new_chunks)} semantic chunks")
        
        print(f"\nFirst 5 old vs new chunks:")
        for i in range(min(5, len(old_lines), len(new_chunks))):
            print(f"\nOld Line {i+1}: {old_lines[i][:80]}...")
            if i < len(new_chunks):
                print(f"New Chunk {i+1}: {new_chunks[i][:80]}...")
        
        # Test the new chunking approach
        total_found = {}
        
        for i, chunk in enumerate(new_chunks[:15], 1):  # Test first 15 chunks
            print(f"\nChunk {i}: {chunk}")
            
            results = self.collection.query(
                query_texts=[chunk],
                n_results=10,
                include=['documents', 'metadatas', 'distances']
            )
            
            chunk_terms = []
            for doc, metadata, distance in zip(
                results['documents'][0], 
                results['metadatas'][0], 
                results['distances'][0]
            ):
                chinese_term = doc
                english_term = metadata['english_term']
                similarity = 1.0 - distance
                
                if similarity >= 0.1:  # Apply threshold
                    chunk_terms.append((chinese_term, english_term, similarity))
                    if chinese_term not in total_found:
                        total_found[chinese_term] = (english_term, similarity)
                    elif similarity > total_found[chinese_term][1]:
                        total_found[chinese_term] = (english_term, similarity)
            
            if chunk_terms:
                print(f"  Found {len(chunk_terms)} terms:")
                for chinese_term, english_term, similarity in chunk_terms:
                    print(f"    {chinese_term} → {english_term} ({similarity:.3f})")
            else:
                print(f"  No terms found above threshold")
        
        print(f"\nTotal unique terms found with new chunking:")
        for chinese_term, (english_term, similarity) in sorted(total_found.items()):
            print(f"  {chinese_term} → {english_term} ({similarity:.3f})")
    
    def test_sentence_splitting_demo(self):
        """Demonstrate how sentence splitting affects the problematic line"""
        
        print(f"\nSentence splitting demonstration ({self.model_name})")
        print("=" * 60)
        
        # The problematic line 2
        problematic_line = '"我是傲视天下，睥睨九霄的绝世丹帝——龙尘？我是人见人欺，无法修行的窝囊废——龙尘？"'
        
        print(f"Original line: {problematic_line}")
        print(f"Length: {len(problematic_line)} chars")
        
        # Test original line
        print(f"\n--- Testing ORIGINAL line ---")
        results = self.collection.query(
            query_texts=[problematic_line],
            n_results=10,
            include=['documents', 'metadatas', 'distances']
        )
        
        for doc, metadata, distance in zip(
            results['documents'][0], 
            results['metadatas'][0], 
            results['distances'][0]
        ):
            chinese_term = doc
            english_term = metadata['english_term']
            similarity = 1.0 - distance
            threshold_status = "✅" if similarity >= 0.1 else "❌"
            print(f"  {chinese_term} → {english_term} (sim: {similarity:.3f}) {threshold_status}")
        
        # Split into sentences
        sentences = []
        current_sentence = ""
        for char in problematic_line:
            current_sentence += char
            if char in ['？', '?']:
                sentence = current_sentence.strip().strip('"').strip('"')
                if len(sentence) >= 5:
                    sentences.append(sentence)
                current_sentence = ""
        
        print(f"\n--- Split into {len(sentences)} sentences ---")
        for i, sentence in enumerate(sentences, 1):
            print(f"\nSentence {i}: {sentence}")
            
            results = self.collection.query(
                query_texts=[sentence],
                n_results=5,
                include=['documents', 'metadatas', 'distances']
            )
            
            for doc, metadata, distance in zip(
                results['documents'][0], 
                results['metadatas'][0], 
                results['distances'][0]
            ):
                chinese_term = doc
                english_term = metadata['english_term']
                similarity = 1.0 - distance
                threshold_status = "✅" if similarity >= 0.1 else "❌"
                print(f"    {chinese_term} → {english_term} (sim: {similarity:.3f}) {threshold_status}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test ChromaDB RAG Retrieval")
    parser.add_argument("--bge", action="store_true", help="Use BGE-M3 embeddings")
    parser.add_argument("--no-qwen", action="store_true", help="Use basic embeddings instead of Qwen3")
    parser.add_argument("--test", choices=["specific", "threshold", "database", "chapter", "sentence"], 
                        default="all", help="Which test to run")
    
    args = parser.parse_args()
    
    print("RAG Retrieval Test")
    print("=" * 60)
    print("Testing ChromaDB semantic search and term retrieval")
    print()
    
    try:
        if args.bge:
            tester = RAGRetrievalTester(use_qwen3=False, use_bge=True)
        else:
            tester = RAGRetrievalTester(use_qwen3=not args.no_qwen, use_bge=False)
        
        if args.test == "sentence":
            tester.test_sentence_splitting_demo()
            return
        
        if args.test == "specific" or args.test == "all":
            tester.test_specific_queries()
        
        if args.test == "threshold" or args.test == "all":
            tester.test_threshold_sensitivity()
        
        if args.test == "database" or args.test == "all":
            tester.test_all_database_terms()
        
        if args.test == "chapter" or args.test == "all":
            tester.test_line_by_line_chapter1()
            
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()