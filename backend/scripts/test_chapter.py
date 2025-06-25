import asyncio
import time
import json
import os
import argparse
from pathlib import Path
from typing import Dict, List
import chromadb
from chromadb.utils import embedding_functions
from chromadb.utils.embedding_functions import EmbeddingFunction
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

def chunk_chinese_text_by_semantic_units(text):
    major_punctuation = ['？', '。', '！', '；', '：', '?', '.', '!', ';', ':']
    minor_punctuation = ['"', '"', '——', '（', '）', '(', ')']
    
    lines = text.split('\n')
    semantic_units = []
    
    for line in lines:
        line = line.strip()
        if len(line) < 5 or line.startswith('第') or line.startswith('﻿'):
            continue
            
        current_units = [line]
        for punct in major_punctuation:
            new_units = []
            for unit in current_units:
                parts = unit.split(punct)
                for part in parts:
                    for minor in minor_punctuation:
                        part = part.replace(minor, ' ')
                    part = part.strip()
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
    def __init__(self, use_bge=True):
        self.use_bge = use_bge
        
        if use_bge:
            self.db_path = "../data/terminology/chroma_db_bge"
            self.collection_name = "bge_terminology"
        else:
            self.db_path = "../data/terminology/chroma_db"
            self.collection_name = "basic_terminology"
        
        self.load_database()
    
    def load_database(self):
        if not Path(self.db_path).exists():
            raise FileNotFoundError(f"ChromaDB database not found: {self.db_path}")
        
        print(f"Loading trained RAG database: {self.db_path}")
        
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        if self.use_bge:
            try:
                self.embedding_function = BGEEmbeddingFunction("BAAI/bge-m3")
                print(f"Using BGE-M3 embeddings")
            except Exception as e:
                print(f"Failed to load BGE model: {e}")
                self._fallback_to_basic()
        else:
            self._fallback_to_basic()
        
        try:
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            
            count = self.collection.count()
            print(f"RAG database loaded: {count} terminology mappings available")
            
        except Exception as e:
            print(f"Error loading collection: {e}")
            raise
    
    def _fallback_to_basic(self):
        print("Using basic multilingual embeddings")
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
    
    def query_terminology(self, chinese_text: str, max_results: int = 10, similarity_threshold: float = 0.15) -> Dict[str, str]:
        if not chinese_text.strip():
            return {}
        
        try:
            semantic_units = chunk_chinese_text_by_semantic_units(chinese_text)
            print(f"Split text into {len(semantic_units)} semantic units for RAG query")
            
            all_terminology = {}
            
            for unit in semantic_units:
                if len(unit.strip()) < 5:
                    continue
                    
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
                    
                    if similarity >= similarity_threshold:
                        if chinese_term not in all_terminology or similarity > all_terminology[chinese_term][1]:
                            all_terminology[chinese_term] = (english_term, similarity)
            
            final_terminology = {chinese: english for chinese, (english, _) in all_terminology.items()}
            
            print(f"Retrieved {len(final_terminology)} terminology mappings from trained RAG")
            return final_terminology
            
        except Exception as e:
            print(f"Error in RAG query: {e}")
            return {}

class ChapterTester:
    def __init__(self, chapter_num: int):
        self.chapter_num = chapter_num
        
        self.chinese_file = f"../data/chapters/clean/chapter_{chapter_num:04d}_cn.txt"
        self.output_file = f"../results/test/chapter_{chapter_num:04d}_test.txt"
        
        Path("../results/test").mkdir(exist_ok=True, parents=True)
        
        self.client = AsyncOpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
        
        self.load_trained_rules()
        self.rag = ChromaRAGQuerySystem(use_bge=True)
    
    def load_trained_rules(self):
        rules_file = "../data/rules/cleaned.json"
        if not Path(rules_file).exists():
            raise FileNotFoundError(f"Trained rules not found: {rules_file}. Run the training pipeline first.")
        
        with open(rules_file, 'r', encoding='utf-8') as f:
            rules_data = json.load(f)
        
        self.rules = []
        for rule_obj in rules_data.get("rules", []):
            self.rules.append(rule_obj["description"])
        
        print(f"Loaded {len(self.rules)} trained style rules")
    
    def load_chapter(self) -> str:
        if not Path(self.chinese_file).exists():
            raise FileNotFoundError(f"Chapter file not found: {self.chinese_file}")
        
        with open(self.chinese_file, 'r', encoding='utf-8') as f:
            chinese_text = f.read().strip()
        
        print(f"Chapter {self.chapter_num} loaded: {len(chinese_text)} characters")
        return chinese_text
    
    async def translate_with_trained_pipeline(self, chinese_text: str, terminology: Dict[str, str]) -> str:
        rules_text = "\n".join([f"- {rule}" for rule in self.rules])
        
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

        try:
            print(f"Translating chapter {self.chapter_num} with trained pipeline...")
            
            response = await self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=1.3,
                max_tokens=8192,
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Translation error: {e}")
            return f"Translation failed: {e}"
    
    async def test_chapter(self):
        print(f"Testing Chapter {self.chapter_num} with Trained Pipeline")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            chinese_text = self.load_chapter()
            
            print("Querying trained RAG database...")
            rag_start = time.time()
            terminology = self.rag.query_terminology(chinese_text)
            rag_elapsed = time.time() - rag_start
            
            if terminology:
                print(f"Found {len(terminology)} RAG mappings in {rag_elapsed:.1f}s")
                print(f"Sample mappings: {dict(list(terminology.items())[:3])}")
            else:
                print("No RAG mappings found - using rules only")
            
            translation = await self.translate_with_trained_pipeline(chinese_text, terminology)
            
            with open(self.output_file, 'w', encoding='utf-8') as f:
                f.write(f"Chapter {self.chapter_num} - Trained Pipeline Translation\n")
                f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"RAG mappings used: {len(terminology)}\n")
                f.write(f"Style rules applied: {len(self.rules)}\n")
                f.write("=" * 60 + "\n\n")
                f.write(translation)
            
            elapsed = time.time() - start_time
            
            print(f"\nChapter {self.chapter_num} Test Complete:")
            print(f"  Translation time: {elapsed:.1f}s")
            print(f"  RAG mappings used: {len(terminology)}")
            print(f"  Style rules applied: {len(self.rules)}")
            print(f"  Output length: {len(translation)} characters")
            print(f"  Saved to: {self.output_file}")
            
            return {
                "chapter": self.chapter_num,
                "success": True,
                "translation_time": elapsed,
                "rag_mappings": len(terminology),
                "output_length": len(translation),
                "output_file": self.output_file
            }
            
        except Exception as e:
            print(f"Error testing chapter {self.chapter_num}: {e}")
            return {
                "chapter": self.chapter_num,
                "success": False,
                "error": str(e)
            }

async def main():
    parser = argparse.ArgumentParser(description="Test Arbitrary Chapter with Trained Pipeline")
    parser.add_argument("--chapter", type=int, required=True, help="Chapter number to test")
    
    args = parser.parse_args()
    
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("Error: DEEPSEEK_API_KEY not found in environment")
        return
    
    required_assets = [
        "../data/rules/cleaned.json",
        "../data/terminology/chroma_db_bge"
    ]
    
    missing_assets = []
    for asset_path in required_assets:
        if not Path(asset_path).exists():
            missing_assets.append(asset_path)
    
    if missing_assets:
        print("Error: Required trained assets not found:")
        for asset_path in missing_assets:
            print(f"  - {asset_path}")
        print("Please run the training pipeline first (steps 1-4)")
        return
    
    tester = ChapterTester(args.chapter)
    result = await tester.test_chapter()
    
    if result["success"]:
        print(f"Successfully tested chapter {args.chapter} with trained pipeline.")
    else:
        print(f"Failed to test chapter {args.chapter}: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    asyncio.run(main())