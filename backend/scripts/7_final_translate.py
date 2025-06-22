import asyncio
import time
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import chromadb
from chromadb.utils import embedding_functions
from chromadb.utils.embedding_functions import EmbeddingFunction
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load environment
load_dotenv()

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
                self.collection_name = "basic_terminology"
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
    
    def query_terminology(self, chinese_text: str, max_results: int = 10, similarity_threshold: float = 0.3) -> Dict[str, str]:
        """
        Query ChromaDB for terminology in Chinese text using semantic similarity
        
        Args:
            chinese_text: Input Chinese text (sentence/paragraph)
            max_results: Maximum number of similar terms to return
            similarity_threshold: Minimum similarity score (0-1)
        
        Returns:
            Dict mapping Chinese terms to English translations
        """
        if not chinese_text.strip():
            return {}
        
        try:
            # Query ChromaDB with the entire Chinese text
            # ChromaDB will find semantically similar Chinese terms in our database
            results = self.collection.query(
                query_texts=[chinese_text],
                n_results=max_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            found_terminology = {}
            
            # Process results
            for doc, metadata, distance in zip(
                results['documents'][0], 
                results['metadatas'][0], 
                results['distances'][0]
            ):
                chinese_term = doc
                english_term = metadata['english_term']
                
                # Convert distance to similarity (ChromaDB uses cosine distance, lower = more similar)
                similarity = 1.0 - distance
                
                # Apply similarity threshold
                if similarity >= similarity_threshold:
                    found_terminology[chinese_term] = english_term
                    print(f"Found: {chinese_term} → {english_term} (similarity: {similarity:.3f})")
            
            print(f"Retrieved {len(found_terminology)} terminology mappings")
            return found_terminology
            
        except Exception as e:
            print(f"Error querying ChromaDB: {e}")
            return {}
    
    def query_chapter(self, chinese_text: str) -> Dict[str, str]:
        return self.query_terminology(chinese_text)

class AsyncFinalTranslator:
    def __init__(self, chapter_num: int, rules: List[str], shared_rag=None):
        self.chapter_num = chapter_num
        self.rules = rules
        
        # File paths
        self.chinese_file = f"../data/chapters/chinese/chapter_{chapter_num:04d}_cn.txt"
        self.ground_truth_file = f"../data/chapters/english/chapter_{chapter_num:04d}_en.txt"
        self.output_file = f"../results/final/translations/chapter_{chapter_num:04d}_final.txt"
        
        # Setup output directory
        Path("../results/final/translations").mkdir(exist_ok=True, parents=True)
        
        # Use shared RAG instance
        self.rag = shared_rag
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
    
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
    
    async def translate_with_rag_and_rules(self, chinese_text: str, terminology: Dict[str, str]) -> str:
        # Build prompt with rules and terminology
        rules_text = "\n".join([f"- {rule}" for rule in self.rules])
        
        terminology_text = ""
        if terminology:
            terminology_text = "\nTERMINOLOGY MAPPINGS (use these exact translations):\n"
            for chinese_term, english_term in terminology.items():
                terminology_text += f"- {chinese_term} = {english_term}\n"
        
        prompt = f"""You are translating a Chinese cultivation novel. Your task is to translate the following Chinese chapter to English with perfect terminology consistency and style.

Follow these style rules and use the provided terminology mappings exactly.

STYLE RULES:
{rules_text}
{terminology_text}
CHINESE TEXT:
{chinese_text}

Please provide a high-quality English translation following the rules and terminology above."""

        try:
            response = await self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=1.3,
                max_tokens=8192,
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Translation error: {e}")
            return f"Translation failed: {e}"
    
    async def process_chapter_async(self, semaphore):
        """Process a single chapter with RAG and rules"""
        async with semaphore:
            print(f"Starting Chapter {self.chapter_num} final translation (Rules + ChromaRAG)")
            
            start_time = time.time()
            
            # Load input files
            try:
                chinese_text, ground_truth = self.load_chapter_files()
                print(f"Chapter {self.chapter_num}: Loaded files - Chinese: {len(chinese_text)} chars")
            except FileNotFoundError as e:
                print(f"Chapter {self.chapter_num}: Error loading: {e}")
                return None
            
            # Query ChromaDB for relevant terminology
            print(f"Chapter {self.chapter_num}: Querying ChromaDB for terminology")
            terminology = self.rag.query_terminology(chinese_text, max_results=15, similarity_threshold=0.4)
            print(f"Chapter {self.chapter_num}: Found {len(terminology)} RAG mappings")
            
            if terminology:
                print(f"Chapter {self.chapter_num}: Sample terminology: {dict(list(terminology.items())[:3])}")
            
            # Translate with rules and RAG
            print(f"Chapter {self.chapter_num}: Translating with rules and RAG terminology")
            translation = await self.translate_with_rag_and_rules(chinese_text, terminology)
            
            # Save translation
            with open(self.output_file, 'w', encoding='utf-8') as f:
                f.write(translation)
            
            elapsed = time.time() - start_time
            print(f"Chapter {self.chapter_num}: Completed in {elapsed:.1f}s")
            print(f"Chapter {self.chapter_num}: Output length: {len(translation)} chars")
            print(f"Chapter {self.chapter_num}: Saved to: {self.output_file}")
            
            return {
                "chapter": self.chapter_num,
                "success": True,
                "output_file": self.output_file,
                "chinese_length": len(chinese_text),
                "translation_length": len(translation),
                "terminology_count": len(terminology),
                "elapsed_time": elapsed
            }

async def translate_chapters_with_rag(start_chapter: int, end_chapter: int, max_concurrent: int = 3, use_qwen3: bool = True):
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
    
    print(f"Loaded {len(rules)} style rules from step 3")
    
    # Create shared RAG system
    print("Initializing shared RAG system...")
    shared_rag = ChromaRAGQuerySystem(use_qwen3=use_qwen3)
    
    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Create translators for each chapter
    translators = []
    for chapter_num in range(start_chapter, end_chapter + 1):
        translator = AsyncFinalTranslator(chapter_num, rules, shared_rag=shared_rag)
        translators.append(translator)
    
    # Process chapters concurrently
    print(f"Processing chapters {start_chapter}-{end_chapter} with {max_concurrent} concurrent requests")
    tasks = [translator.process_chapter_async(semaphore) for translator in translators]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    successful = 0
    total_time = 0
    total_terminology = 0
    
    for result in results:
        if isinstance(result, dict) and result.get("success"):
            successful += 1
            total_time += result["elapsed_time"]
            total_terminology += result["terminology_count"]
        else:
            print(f"Failed result: {result}")
    
    print(f"\nFinal Translation Complete")
    print(f"Successful chapters: {successful}/{len(translators)}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average time per chapter: {total_time/len(translators):.1f}s")
    print(f"Average terminology per chapter: {total_terminology/len(translators):.1f}")
    print(f"Output directory: ../results/final/translations/")

def test_chroma_rag_system():
    """Test the ChromaDB RAG system"""
    print("Testing ChromaDB RAG System")
    print("=" * 50)
    
    try:
        rag = ChromaRAGQuerySystem(use_qwen3=True)
        
        # Test queries
        test_texts = [
            "龙尘突破到金丹期，成为了丹帝传人",
            "他修炼九星霸体诀，实力大增",
            "筑基期的修士都很强大"
        ]
        
        for text in test_texts:
            print(f"\nQuery: {text}")
            results = rag.query_terminology(text, max_results=5, similarity_threshold=0.3)
            if results:
                for chinese, english in results.items():
                    print(f"  {chinese} → {english}")
            else:
                print("  No matches found")
                
    except Exception as e:
        print(f"Error testing RAG system: {e}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Step 7: Final Translation with ChromaDB RAG")
    parser.add_argument("--start", type=int, default=1, help="Start chapter number")
    parser.add_argument("--end", type=int, default=3, help="End chapter number")
    parser.add_argument("--concurrent", type=int, default=3, help="Max concurrent requests")
    parser.add_argument("--test", action="store_true", help="Test RAG system only")
    parser.add_argument("--no-qwen", action="store_true", help="Use basic embeddings instead of Qwen3")
    
    args = parser.parse_args()
    
    print("Step 7: Final Translation with ChromaDB RAG + Style Rules")
    print("=" * 60)
    
    if args.test:
        test_chroma_rag_system()
        return
    
    # Check API key
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("Error: DEEPSEEK_API_KEY not found in environment")
        return
    
    use_qwen3 = not args.no_qwen
    model_type = "Qwen3" if use_qwen3 else "basic sentence-transformers"
    print(f"Using {model_type} embeddings for RAG")
    
    # Run translation
    try:
        asyncio.run(translate_chapters_with_rag(
            start_chapter=args.start,
            end_chapter=args.end,
            max_concurrent=args.concurrent,
            use_qwen3=use_qwen3
        ))
    except KeyboardInterrupt:
        print("\nTranslation interrupted by user")
    except Exception as e:
        print(f"Translation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()