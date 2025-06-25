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
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

load_dotenv()


def chunk_chinese_text_by_semantic_units(text):
    # Major punctuation that creates semantic boundaries
    major_punctuation = ['？', '。', '！', '；', '：', '?', '.', '!', ';', ':']
    # Minor punctuation to clean from boundaries  
    minor_punctuation = ['"', '"', '——', '（', '）', '(', ')']
    
    lines = text.split('\n')
    semantic_units = []
    
    for line in lines:
        line = line.strip()
        if len(line) < 5 or line.startswith('第') or line.startswith('﻿'):
            continue
            
        # Split by major punctuation to create focused semantic units
        current_units = [line]
        for punct in major_punctuation:
            new_units = []
            for unit in current_units:
                parts = unit.split(punct)
                for part in parts:
                    # Clean up minor punctuation and whitespace
                    for minor in minor_punctuation:
                        part = part.replace(minor, ' ')
                    part = part.strip()
                    
                    # Only keep substantial semantic units
                    if len(part) >= 5:
                        new_units.append(part)
            current_units = new_units
        
        semantic_units.extend(current_units)
    
    return semantic_units


class BGEEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name="BAAI/bge-m3"):
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
    
    def __call__(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

class ChromaRAGQuerySystem:
    def __init__(self, use_qwen3=False, use_bge=True, qwen_model="Qwen/Qwen3-Embedding-8B"):
        self.use_qwen3 = use_qwen3
        self.use_bge = use_bge
        self.qwen_model = qwen_model
        
        # Set database path based on embedding choice
        if use_bge:
            self.db_path = "../data/terminology/chroma_db_bge"
            self.collection_name = "bge_terminology"
        elif use_qwen3:
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
        
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        # Setup embedding function based on configuration
        if self.use_bge:
            try:
                self.embedding_function = BGEEmbeddingFunction("BAAI/bge-m3")
                print(f"Using BGE-M3 embeddings")
            except Exception as e:
                print(f"Failed to load BGE model: {e}")
                self._fallback_to_basic()
        elif self.use_qwen3:
            try:
                from sentence_transformers import SentenceTransformer
                class Qwen3EmbeddingFunction(EmbeddingFunction):
                    def __init__(self, model_name):
                        import torch
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        self.model = SentenceTransformer(model_name, device=device)
                    def __call__(self, texts: List[str]) -> List[List[float]]:
                        embeddings = self.model.encode(texts, convert_to_numpy=True)
                        return embeddings.tolist()
                
                self.embedding_function = Qwen3EmbeddingFunction(self.qwen_model)
                print(f"Using Qwen3 embeddings: {self.qwen_model}")
            except Exception as e:
                print(f"Failed to load Qwen3 model: {e}")
                self._fallback_to_basic()
        else:
            self._fallback_to_basic()
        
        # Load collection
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
    
    def _fallback_to_basic(self):
        print("Falling back to basic multilingual embeddings")
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
    
    def query_terminology_parallel(self, chinese_text: str, max_results: int = 10, similarity_threshold: float = 0.15, max_workers: int = 8) -> Dict[str, str]:
        if not chinese_text.strip():
            return {}
        
        try:
            # Use improved semantic chunking
            semantic_units = chunk_chinese_text_by_semantic_units(chinese_text)
            print(f"Split text into {len(semantic_units)} semantic units for RAG query")
            
            all_terminology = {}
            
            # Process semantic units in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_unit = {
                    executor.submit(self._query_single_unit, i, unit, max_results, similarity_threshold): (i, unit)
                    for i, unit in enumerate(semantic_units) if len(unit.strip()) >= 5
                }
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_unit):
                    i, unit = future_to_unit[future]
                    try:
                        unit_terminology = future.result()
                        
                        # Merge results, keeping highest similarity for duplicates
                        for chinese_term, (english_term, similarity) in unit_terminology.items():
                            if chinese_term not in all_terminology or similarity > all_terminology[chinese_term][1]:
                                all_terminology[chinese_term] = (english_term, similarity)
                                
                    except Exception as e:
                        print(f"Error processing unit {i+1}: {e}")
                        continue
            
            # Convert to final format (remove similarity scores)
            final_terminology = {chinese: english for chinese, (english, _) in all_terminology.items()}
            
            print(f"Retrieved {len(final_terminology)} terminology mappings total (parallel processing)")
            return final_terminology
            
        except Exception as e:
            print(f"Error in parallel ChromaDB query: {e}")
            return {}
    
    def _query_single_unit(self, unit_index: int, unit: str, max_results: int, similarity_threshold: float) -> Dict[str, Tuple[str, float]]:
        unit_terminology = {}
        
        try:
            results = self.collection.query(
                query_texts=[unit],
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
                
                if similarity >= 0.15:  # Lower threshold to catch context-diluted terms
                    unit_terminology[chinese_term] = (english_term, similarity)
                    
        except Exception as e:
            # Don't print here to avoid thread collision in logs
            pass
            
        return unit_terminology

class AsyncFinalTranslator:
    def __init__(self, chapter_num: int, rules: List[str], shared_rag=None, debug=False, dry_run=False, start_chapter=1):
        self.chapter_num = chapter_num
        self.rules = rules
        self.debug = debug
        self.dry_run = dry_run
        self.start_chapter = start_chapter  # Added for progress bar positioning
        
        # File paths
        self.chinese_file = f"../data/chapters/clean/chapter_{chapter_num:04d}_cn.txt"
        self.ground_truth_file = f"../data/chapters/ground_truth/chapter_{chapter_num:04d}_en.txt"
        self.output_file = f"../results/final/translations/chapter_{chapter_num:04d}_final.txt"
        self.debug_file = f"../debug/prompts/chapter_{chapter_num:04d}_final_prompt.txt"
        
        # Setup output directories
        Path("../results/final/translations").mkdir(exist_ok=True, parents=True)
        if self.debug or self.dry_run:
            Path("../debug/prompts").mkdir(exist_ok=True, parents=True)
        
        # Use shared RAG instance
        self.rag = shared_rag
        
        # Initialize OpenAI client only if not dry run
        if not self.dry_run:
            self.client = AsyncOpenAI(
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com"
            )

    def load_chapter_files(self) -> Tuple[str, str]:
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
        # Build style rules text
        rules_text = "\n".join([f"- {rule}" for rule in self.rules])
        
        # Build terminology examples (not exact mappings)
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

        # Save debug information if enabled
        if self.debug or self.dry_run:
            debug_content = f"""=== CHAPTER {self.chapter_num} FINAL TRANSLATION PROMPT ===
Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}
Total terminology found: {len(terminology)}
Mode: {'DRY RUN' if self.dry_run else 'LIVE TRANSLATION'}

=== TERMINOLOGY RETRIEVED ===
{json.dumps(terminology, indent=2, ensure_ascii=False)}

=== SYSTEM MESSAGE ===
{system_message}

=== USER PROMPT ===
{user_prompt}

=== PROMPT STATS ===
System message length: {len(system_message)} chars
User prompt length: {len(user_prompt)} chars
Total prompt length: {len(system_message) + len(user_prompt)} chars
Terminology count: {len(terminology)}

=== END DEBUG ===
"""
        
            with open(self.debug_file, 'w', encoding='utf-8') as f:
                f.write(debug_content)
            
            print(f"Chapter {self.chapter_num}: Debug prompt saved to {self.debug_file}")

        # Return early if dry run
        if self.dry_run:
            return f"[DRY RUN] Prompt constructed successfully for Chapter {self.chapter_num}"

        try:
            # Estimate tokens for progress bar
            estimated_total_tokens = int(len(chinese_text) * 1.31)
            
            # Initialize progress bar for this chapter (fixed positioning)
            with tqdm(
                total=estimated_total_tokens,
                desc=f"Chapter {self.chapter_num}",
                unit="tok",
                position=self.chapter_num - self.start_chapter,  # Fixed positioning
                leave=True
            ) as pbar:
                
                # Enable streaming
                response = await self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=1.3,
                    max_tokens=8192,
                    stream=True
                )
                
                # Accumulate translation with live progress
                translation = ""
                tokens_received = 0
                
                async for chunk in response:
                    if chunk.choices[0].delta.content:
                        new_content = chunk.choices[0].delta.content
                        translation += new_content
                        
                        # Count tokens (rough: words + punctuation)
                        new_tokens = len(new_content.split()) + new_content.count(',') + new_content.count('.')
                        tokens_received += new_tokens
                        
                        # Update progress bar
                        pbar.update(new_tokens)
                
                # Ensure progress bar reaches 100%
                if tokens_received < estimated_total_tokens:
                    pbar.update(estimated_total_tokens - tokens_received)
            
            return translation.strip()
            
        except Exception as e:
            print(f"Translation error: {e}")
            return f"Translation failed: {e}"
    
    async def process_chapter_async(self, semaphore):
        async with semaphore:
            start_time = time.time()
            
            try:
                mode = "DRY RUN" if self.dry_run else "LIVE"
                print(f"Starting Chapter {self.chapter_num} final translation ({mode} - Rules + Parallel ChromaRAG)")
                
                # Load input files
                try:
                    chinese_text, ground_truth = self.load_chapter_files()
                    print(f"Chapter {self.chapter_num}: Loaded files - Chinese: {len(chinese_text)} chars")
                except FileNotFoundError as e:
                    print(f"Chapter {self.chapter_num}: Error loading: {e}")
                    return None
                
                # Query ChromaDB for relevant terminology using parallel semantic chunking
                print(f"Chapter {self.chapter_num}: Querying ChromaDB for terminology (parallel processing)")
                rag_start = time.time()
                
                # Run RAG query in thread pool to avoid blocking async loop
                loop = asyncio.get_event_loop()
                terminology = await loop.run_in_executor(
                    None, 
                    self.rag.query_terminology_parallel, 
                    chinese_text, 
                    10,  # max_results
                    0.2, # similarity_threshold  
                    8    # max_workers
                )
                
                rag_elapsed = time.time() - rag_start
                print(f"Chapter {self.chapter_num}: Found {len(terminology)} RAG mappings in {rag_elapsed:.1f}s")
                
                if terminology:
                    print(f"Chapter {self.chapter_num}: Sample terminology: {dict(list(terminology.items())[:3])}")
                
                # Translate with rules and RAG
                print(f"Chapter {self.chapter_num}: Translating with rules and RAG terminology")
                translation = await self.translate_with_rag_and_rules(chinese_text, terminology)
                
                # Save actual translation only if not dry run
                if not self.dry_run:
                    with open(self.output_file, 'w', encoding='utf-8') as f:
                        f.write(translation)
                    print(f"Chapter {self.chapter_num}: Saved to: {self.output_file}")
                else:
                    print(f"Chapter {self.chapter_num}: Dry run complete - no translation file saved")
                
                elapsed = time.time() - start_time
                print(f"Chapter {self.chapter_num}: Completed in {elapsed:.1f}s (RAG: {rag_elapsed:.1f}s)")
                print(f"Chapter {self.chapter_num}: Output length: {len(translation)} chars")
                
                return {
                    "chapter": self.chapter_num,
                    "success": True,
                    "output_file": self.output_file,
                    "chinese_length": len(chinese_text),
                    "translation_length": len(translation),
                    "terminology_count": len(terminology),
                    "elapsed_time": elapsed,
                    "rag_time": rag_elapsed
                }

            except Exception as e:
                print(f"Error processing Chapter {self.chapter_num}: {e}")
                return {
                    "chapter": self.chapter_num,
                    "success": False,
                    "error": str(e)
                }

async def translate_chapters_with_rag(start_chapter: int, end_chapter: int, max_concurrent: int = 3, use_qwen3: bool = False, use_bge: bool = True, debug: bool = False, dry_run: bool = False):
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
    print("Initializing shared RAG system")
    shared_rag = ChromaRAGQuerySystem(use_qwen3=use_qwen3, use_bge=use_bge)
    
    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Create translators for each chapter
    translators = []
    for chapter_num in range(start_chapter, end_chapter + 1):
        translator = AsyncFinalTranslator(
            chapter_num, 
            rules, 
            shared_rag=shared_rag, 
            debug=debug, 
            dry_run=dry_run,
            start_chapter=start_chapter  # Pass start_chapter for progress bar positioning
        )
        translators.append(translator)
    
    # Process chapters concurrently
    mode = "DRY RUN" if dry_run else "LIVE TRANSLATION"
    print(f"Processing chapters {start_chapter}-{end_chapter} with {max_concurrent} concurrent requests ({mode})")
    if debug or dry_run:
        print("Debug mode enabled - prompts will be saved to ../debug/prompts/")
    if dry_run:
        print("DRY RUN MODE - No API calls will be made")
    
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
    
    print(f"\n{mode} Complete")
    print(f"Successful chapters: {successful}/{len(translators)}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average time per chapter: {total_time/len(translators):.1f}s")
    print(f"Average terminology per chapter: {total_terminology/len(translators):.1f}")
    if not dry_run:
        print(f"Output directory: ../results/final/translations/")
    if debug or dry_run:
        print(f"Debug prompts saved to: ../debug/prompts/")

def test_chroma_rag_system():
    print("Testing ChromaDB RAG System")
    print("=" * 50)
    
    try:
        rag = ChromaRAGQuerySystem(use_qwen3=False, use_bge=True)
        
        # Test queries
        test_texts = [
            "龙尘突破到金丹期，成为了丹帝传人",
            "他修炼九星霸体诀，实力大增",
            "筑基期的修士都很强大"
        ]
        
        for text in test_texts:
            print(f"\nQuery: {text}")
            results = rag.query_terminology_parallel(text, max_results=5, similarity_threshold=0.3)
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
    parser.add_argument("--qwen", action="store_true", help="Use Qwen3-8B embeddings instead of BGE-M3")
    parser.add_argument("--lite", action="store_true", help="Use basic embeddings instead of BGE-M3")
    parser.add_argument("--debug", action="store_true", help="Save full prompts to debug folder")
    parser.add_argument("--dry-run", action="store_true", help="Build prompts without sending to API")
    
    args = parser.parse_args()
    
    print("Step 7: Final Translation with ChromaDB RAG + Style Rules")
    print("=" * 60)
    
    if args.test:
        test_chroma_rag_system()
        return
    
    # Check API key unless dry run
    if not args.dry_run and not os.getenv("DEEPSEEK_API_KEY"):
        print("Error: DEEPSEEK_API_KEY not found in environment")
        return
    
    # Determine embedding strategy
    if args.qwen:
        use_qwen3 = True
        use_bge = False
    elif args.lite:
        use_qwen3 = False
        use_bge = False
    else:
        use_qwen3 = False
        use_bge = True

    model_type = "BGE-M3" if use_bge else ("Qwen3" if use_qwen3 else "basic sentence-transformers")
    print(f"Using {model_type} embeddings for RAG")
    
    # Run translation
    try:
        asyncio.run(translate_chapters_with_rag(
            start_chapter=args.start,
            end_chapter=args.end,
            max_concurrent=args.concurrent,
            use_qwen3=use_qwen3,
            use_bge=use_bge,
            debug=args.debug,
            dry_run=args.dry_run
        ))
    except KeyboardInterrupt:
        print("\nTranslation interrupted by user")
    except Exception as e:
        print(f"Translation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()