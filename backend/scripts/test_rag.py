import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import time
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import chromadb
from chromadb.utils import embedding_functions
from chromadb.utils.embedding_functions import EmbeddingFunction
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()


def chunk_chinese_text_by_lines(text):
    lines = text.split('\n')
    
    # Clean up lines
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        # Skip empty lines, chapter headers, and very short lines
        if len(line) >= 5 and not line.startswith('第') and not line.startswith('﻿'):
            cleaned_lines.append(line)
    
    return cleaned_lines

class Qwen3EmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name="Qwen/Qwen3-Embedding-8B"):
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
    
    def __call__(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

class ChromaRAGQuerySystem:
    def __init__(self, use_qwen3=True, qwen_model="Qwen/Qwen3-Embedding-8B"):
        self.use_qwen3 = use_qwen3
        self.qwen_model = qwen_model
        
        if use_qwen3:
            self.db_path = "../data/terminology/chroma_db_rag"
            self.collection_name = "rag_terminology"
        else:
            self.db_path = "../data/terminology/chroma_db"
            self.collection_name = "basic_terminology"
        
        self.load_database()
    
    def load_database(self):
        if not Path(self.db_path).exists():
            raise FileNotFoundError(f"ChromaDB database not found: {self.db_path}")
        
        print(f"Loading ChromaDB from: {self.db_path}")
        
        # Create client
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        # Setup embedding function
        if self.use_qwen3:
            try:
                self.embedding_function = Qwen3EmbeddingFunction(
                    model_name=self.qwen_model
                )
                print(f"Using Qwen3 embeddings: {self.qwen_model}")
            except Exception as e:
                print(f"Failed to load Qwen3 model: {e}")
                print("Falling back to sentence-transformers")
                self.use_qwen3 = False
                self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="paraphrase-multilingual-MiniLM-L12-v2"
                )
        else:
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="paraphrase-multilingual-MiniLM-L12-v2"
            )
        
        # Get collection
        try:
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            
            count = self.collection.count()
            print(f"ChromaDB collection loaded: {self.collection_name}")
            print(f"Total terms available: {count}")
            
        except Exception as e:
            print(f"Error loading collection: {e}")
            print("Available collections:", self.client.list_collections())
            raise
    
    def query_terminology_parallel(self, chinese_text: str, max_results: int = 10, similarity_threshold: float = 0.2, max_workers: int = 8) -> Dict[str, str]:
        if not chinese_text.strip():
            return {}
        
        try:
            # Chunk text into lines
            lines = chunk_chinese_text_by_lines(chinese_text)
            print(f"Split text into {len(lines)} lines for parallel RAG query")
            
            all_terminology = {}
            
            # Process lines in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all line queries concurrently
                future_to_line = {
                    executor.submit(self._query_single_line, i, line, max_results, similarity_threshold): (i, line)
                    for i, line in enumerate(lines) if len(line.strip()) >= 5
                }
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_line):
                    i, line = future_to_line[future]
                    try:
                        line_terminology = future.result()
                        
                        # Merge results, keeping highest similarity for duplicates
                        for chinese_term, (english_term, similarity) in line_terminology.items():
                            if chinese_term not in all_terminology or similarity > all_terminology[chinese_term][1]:
                                all_terminology[chinese_term] = (english_term, similarity)
                                print(f"Found: {chinese_term} → {english_term} (similarity: {similarity:.3f}) [Line {i+1}]")
                                
                    except Exception as e:
                        print(f"Error processing line {i+1}: {e}")
                        continue
            
            # Convert to final format (remove similarity scores)
            final_terminology = {chinese: english for chinese, (english, _) in all_terminology.items()}
            
            print(f"Retrieved {len(final_terminology)} terminology mappings total (parallel processing)")
            return final_terminology
            
        except Exception as e:
            print(f"Error in parallel ChromaDB query: {e}")
            return {}
    
    def _query_single_line(self, line_index: int, line: str, max_results: int, similarity_threshold: float) -> Dict[str, Tuple[str, float]]:
        """
        Query a single line against ChromaDB
        Returns dict of {chinese_term: (english_term, similarity)}
        """
        line_terminology = {}
        
        try:
            results = self.collection.query(
                query_texts=[line],
                n_results=max_results,
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
                
                if similarity >= similarity_threshold:
                    line_terminology[chinese_term] = (english_term, similarity)
                    
        except Exception as e:
            # Don't print here to avoid thread collision in logs
            pass
            
        return line_terminology

class RAGPromptTester:
    def __init__(self, chapter_num: int, rules: List[str], shared_rag=None):
        self.chapter_num = chapter_num
        self.rules = rules
        
        # File paths
        self.chinese_file = f"../data/chapters/clean/chapter_{chapter_num:04d}_cn.txt"
        self.ground_truth_file = f"../data/chapters/ground_truth/chapter_{chapter_num:04d}_en.txt"
        self.prompt_output_file = f"../debug/prompts/chapter_{chapter_num:04d}_prompt.txt"
        
        # Setup output directory
        Path("../debug/prompts").mkdir(exist_ok=True, parents=True)
        
        # Use shared RAG instance
        self.rag = shared_rag
    
    def load_chapter_files(self) -> Tuple[str, str]:
        """Load Chinese text and ground truth files"""
        if not Path(self.chinese_file).exists():
            raise FileNotFoundError(f"Chinese file not found: {self.chinese_file}")
        
        if not Path(self.ground_truth_file).exists():
            raise FileNotFoundError(f"Ground truth file not found: {self.ground_truth_file}")
        
        with open(self.chinese_file, 'r', encoding='utf-8') as f:
            chinese_text = f.read().strip()
        
        with open(self.ground_truth_file, 'r', encoding='utf-8') as f:
            ground_truth = f.read().strip()
        
        return chinese_text, ground_truth
    
    def build_prompt_with_rag_and_rules(self, chinese_text: str, terminology: Dict[str, str]) -> Dict[str, str]:
        # Build style rules text
        rules_text = "\n".join([f"- {rule}" for rule in self.rules])
        
        # Build terminology examples (not exact mappings!)
        terminology_examples = ""
        if terminology:
            terminology_examples = "\nRELEVANT PROFESSIONAL TRANSLATION EXAMPLES:\n"
            terminology_examples += "These are terminology choices from professional translations of similar contexts:\n\n"
            
            for chinese_term, english_term in terminology.items():
                terminology_examples += f"• {chinese_term} → {english_term}\n"
            
            terminology_examples += "\nUse these examples to maintain consistency with professional standards.\n"
        
        system_message = "You are an expert translator. Learn from professional examples to maintain consistency while preserving natural flow."
        
        user_prompt = f"""You are translating a Chinese cultivation novel. Create a natural English translation that learns from professional translation patterns.

STYLE PATTERNS (learned from professional translations):
{rules_text}

{terminology_examples}

CHINESE TEXT:
{chinese_text}

Translate naturally, using the professional examples as guidance for terminology consistency:"""

        return {
            "system": system_message,
            "user": user_prompt
        }
    
    def test_chapter_prompt(self):
        """Test prompt construction for a single chapter - NO AI CALL"""
        print(f"Testing Chapter {self.chapter_num} prompt construction")
        print("=" * 60)
        
        start_time = time.time()
        
        # Load input files
        try:
            chinese_text, ground_truth = self.load_chapter_files()
            print(f"✅ Loaded files - Chinese: {len(chinese_text)} chars, Ground truth: {len(ground_truth)} chars")
        except FileNotFoundError as e:
            print(f"❌ Error loading: {e}")
            return None
        
        # Query ChromaDB for relevant terminology
        print(f"\n🔍 Querying ChromaDB for terminology...")
        rag_start = time.time()
        
        terminology = self.rag.query_terminology_parallel(
            chinese_text, 
            max_results=10,
            similarity_threshold=0.2,
            max_workers=8
        )
        
        rag_elapsed = time.time() - rag_start
        print(f"✅ Found {len(terminology)} RAG mappings in {rag_elapsed:.1f}s")
        
        if terminology:
            print(f"\n📝 Retrieved terminology:")
            for chinese, english in terminology.items():
                print(f"   {chinese} → {english}")
        else:
            print(f"\n⚠️  No terminology found above threshold")
        
        # Build prompt (but don't send to AI)
        print(f"\n🔧 Building prompt...")
        prompt_data = self.build_prompt_with_rag_and_rules(chinese_text, terminology)
        
        # Create debug output
        debug_content = f"""=== CHAPTER {self.chapter_num} PROMPT DEBUG ===
Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}
RAG retrieval time: {rag_elapsed:.1f}s
Total terminology found: {len(terminology)}

=== TERMINOLOGY RETRIEVED ===
{json.dumps(terminology, indent=2, ensure_ascii=False)}

=== SYSTEM MESSAGE ===
{prompt_data['system']}

=== USER PROMPT ===
{prompt_data['user']}

=== END DEBUG ===
"""
        
        # Save to debug file
        with open(self.prompt_output_file, 'w', encoding='utf-8') as f:
            f.write(debug_content)
        
        elapsed = time.time() - start_time
        print(f"\n✅ Prompt test completed in {elapsed:.1f}s")
        print(f"📁 Debug output saved to: {self.prompt_output_file}")
        print(f"📊 Prompt stats:")
        print(f"   - System message: {len(prompt_data['system'])} chars")
        print(f"   - User prompt: {len(prompt_data['user'])} chars")
        print(f"   - Total prompt size: {len(prompt_data['system']) + len(prompt_data['user'])} chars")
        
        return {
            "chapter": self.chapter_num,
            "success": True,
            "terminology_count": len(terminology),
            "elapsed_time": elapsed,
            "rag_time": rag_elapsed,
            "prompt_file": self.prompt_output_file,
            "prompt_size": len(prompt_data['system']) + len(prompt_data['user'])
        }

def test_rag_prompt_construction():
    """Test RAG prompt construction without AI calls"""
    
    # Load style rules from step 3
    rules_file = "../data/rules/cleaned.json"
    if not Path(rules_file).exists():
        raise FileNotFoundError(f"Rules file not found: {rules_file}. Run step 3 first.")
    
    with open(rules_file, 'r', encoding='utf-8') as f:
        rules_data = json.load(f)
    
    # Extract rules
    rules = []
    for rule_obj in rules_data.get("rules", []):
        rules.append(rule_obj["description"])
    
    print(f"✅ Loaded {len(rules)} style rules from step 3")
    
    # Create shared RAG system
    print("🔧 Initializing RAG system...")
    shared_rag = ChromaRAGQuerySystem(use_qwen3=True)
    
    # Test chapters 1-3
    test_chapters = [1, 2, 3]
    results = []
    
    for chapter_num in test_chapters:
        print(f"\n{'='*60}")
        tester = RAGPromptTester(chapter_num, rules, shared_rag=shared_rag)
        result = tester.test_chapter_prompt()
        if result:
            results.append(result)
    
    # Summary
    print(f"\n{'='*60}")
    print("🎯 TEST SUMMARY")
    print(f"{'='*60}")
    
    if results:
        avg_terminology = sum(r["terminology_count"] for r in results) / len(results)
        avg_time = sum(r["elapsed_time"] for r in results) / len(results)
        avg_rag_time = sum(r["rag_time"] for r in results) / len(results)
        avg_prompt_size = sum(r["prompt_size"] for r in results) / len(results)
        
        print(f"✅ Tested {len(results)} chapters successfully")
        print(f"📊 Average terminology per chapter: {avg_terminology:.1f}")
        print(f"⏱️  Average total time: {avg_time:.1f}s")
        print(f"🔍 Average RAG time: {avg_rag_time:.1f}s")
        print(f"📝 Average prompt size: {avg_prompt_size:.0f} chars")
        print(f"📁 Debug files saved to: ../debug/prompts/")
        print(f"\n💡 Review the prompt files to verify RAG retrieval and prompt construction!")
    else:
        print("❌ No chapters tested successfully")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test RAG Prompt Construction")
    parser.add_argument("--chapter", type=int, help="Test specific chapter number")
    parser.add_argument("--no-qwen", action="store_true", help="Use basic embeddings instead of Qwen3")
    
    args = parser.parse_args()
    
    print("🧪 RAG PROMPT CONSTRUCTION TEST")
    print("=" * 60)
    print("This test runs everything EXCEPT the AI call")
    print("It will show you exactly what prompt would be sent to DeepSeek")
    print()
    
    try:
        if args.chapter:
            # Test single chapter
            rules_file = "../data/rules/cleaned.json"
            with open(rules_file, 'r', encoding='utf-8') as f:
                rules_data = json.load(f)
            rules = [rule_obj["description"] for rule_obj in rules_data.get("rules", [])]
            
            shared_rag = ChromaRAGQuerySystem(use_qwen3=not args.no_qwen)
            tester = RAGPromptTester(args.chapter, rules, shared_rag=shared_rag)
            tester.test_chapter_prompt()
        else:
            # Test all chapters
            test_rag_prompt_construction()
            
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()