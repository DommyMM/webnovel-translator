import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from collections import defaultdict, Counter
import chromadb
from chromadb.utils import embedding_functions
from chromadb.utils.embedding_functions import EmbeddingFunction
from sentence_transformers import SentenceTransformer

class BGEEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name="BAAI/bge-m3"):
        print(f"Loading embedding model: {model_name}")
        
        # GPU optimizations
        import torch
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            gpu_name = torch.cuda.get_device_name(0)
            print(f"GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
            device = "cuda"
        else:
            device = "cpu"
            print("No GPU detected, using CPU")
        
        self.model = SentenceTransformer(model_name, device=device)
        
        print(f"Model loaded on {device}")
        print(f"Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def __call__(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

class Qwen3EmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name="Qwen/Qwen3-Embedding-8B"):
        print(f"Loading embedding model: {model_name}")
        
        # GPU optimizations
        import torch
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            gpu_name = torch.cuda.get_device_name(0)
            print(f"GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
            
            if gpu_memory > 20:
                print("High-end GPU detected - using optimized settings")
                device = "cuda"
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = "cpu"
            print("No GPU detected, using CPU")
        
        self.model = SentenceTransformer(model_name, device=device)
        
        print(f"Model loaded on {device}")
        print(f"Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def __call__(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

class ChromaTerminologyBuilder:
    def __init__(self, use_qwen3=False, use_bge=True, qwen_model="Qwen/Qwen3-Embedding-8B"):
        self.input_file = "../data/terminology/extracted_terminology.json"
        self.use_qwen3 = use_qwen3
        self.use_bge = use_bge
        self.qwen_model = qwen_model
        
        if use_bge:
            self.db_path = "../data/terminology/chroma_db_bge"
            self.readable_file = "../data/terminology/bge_terminology_readable.txt"
            self.embedding_model_name = "BAAI/bge-m3"
            self.collection_name = "bge_terminology"
        elif use_qwen3:
            self.db_path = "../data/terminology/chroma_db_rag"
            self.readable_file = "../data/terminology/rag_terminology_readable.txt"
            self.embedding_model_name = qwen_model
            self.collection_name = "rag_terminology"
        else:
            self.db_path = "../data/terminology/chroma_db"
            self.readable_file = "../data/terminology/chroma_terminology_readable.txt"
            self.embedding_model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
            self.collection_name = "basic_terminology"
            
        self.setup_directories()
        self.setup_chromadb()
    
    def setup_directories(self):
        Path("../data/terminology").mkdir(exist_ok=True, parents=True)
        Path(self.db_path).mkdir(exist_ok=True, parents=True)
    
    def setup_chromadb(self):
        if self.use_bge:
            print("Setting up ChromaDB with BGE-M3 embeddings...")
        elif self.use_qwen3:
            print("Setting up ChromaDB with Qwen3 embeddings...")
        else:
            print("Setting up ChromaDB with sentence-transformers...")
        
        # Create persistent client
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        # Choose embedding function
        if self.use_bge:
            try:
                self.embedding_function = BGEEmbeddingFunction("BAAI/bge-m3")
            except Exception as e:
                print(f"Failed to load BGE model: {e}")
                raise
        elif self.use_qwen3:
            try:
                self.embedding_function = Qwen3EmbeddingFunction(
                    model_name=self.qwen_model
                )
            except Exception as e:
                print(f"Failed to load Qwen3 model: {e}")
                print("Falling back to sentence-transformers")
                self.use_qwen3 = False
                self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
                )
                self.embedding_model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
                self.collection_name = "basic_terminology"
        else:
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
            )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={
                "description": "Chinese cultivation novel terminology mappings",
                "created_at": datetime.now().isoformat(),
                "language_pair": "Chinese-English",
                "embedding_model": self.embedding_model_name
            }
        )
        
        print(f"ChromaDB collection ready: {self.collection.name}")
        print(f"Database path: {self.db_path}")
        print(f"Embedding model: {self.embedding_model_name}")
    
    def load_extracted_terminology(self) -> Dict:
        if not Path(self.input_file).exists():
            raise FileNotFoundError(f"Extracted terminology not found: {self.input_file}")
        
        with open(self.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded extracted terminology with {data['metadata']['total_terms']} terms")
        return data
    
    def clean_and_prepare_terminology(self, raw_data: Dict) -> Dict:
        raw_terminology = raw_data.get("terminology", {})
        
        clean_db = {}
        category_counts = Counter()
        frequency_stats = []
        
        for chinese_term, term_data in raw_terminology.items():
            professional_term = term_data["professional_term"]
            category = term_data.get("category", "general")
            frequency = term_data.get("frequency", 1)
            confidence = term_data.get("confidence", 0.8)
            chapters_seen = term_data.get("chapters_seen", [])
            
            # Clean and validate
            chinese_term = chinese_term.strip()
            professional_term = professional_term.strip()
            
            # Skip empty or single character terms
            if not chinese_term or not professional_term or len(chinese_term) < 2:
                continue
            
            clean_entry = {
                "english_term": professional_term,
                "category": category,
                "frequency": frequency,
                "confidence": confidence,
                "chapters_seen": str(chapters_seen) if chapters_seen else "[]",
                "first_seen": min(chapters_seen) if chapters_seen else 1,
                "last_seen": max(chapters_seen) if chapters_seen else 1,
                "created_at": datetime.now().isoformat()
            }
            
            clean_db[chinese_term] = clean_entry
            category_counts[category] += 1
            frequency_stats.append(frequency)
        
        metadata = {
            "created_at": datetime.now().isoformat(),
            "total_terms": len(clean_db),
            "source_method": "ai_extraction",
            "categories": dict(category_counts),
            "avg_frequency": sum(frequency_stats) / len(frequency_stats) if frequency_stats else 0,
            "max_frequency": max(frequency_stats) if frequency_stats else 0,
            "chapters_covered": raw_data["metadata"].get("chapters_processed", "unknown"),
            "embedding_model": self.embedding_model_name
        }
        
        return {
            "metadata": metadata,
            "terminology": clean_db
        }
    def generate_term_id(self, chinese_term: str) -> str:  # Generate consistent ID for a term using hash
        return f"term_{hashlib.md5(chinese_term.encode('utf-8')).hexdigest()}"
    
    def update_vector_database(self, clean_data: Dict):
        terminology = clean_data["terminology"]
        metadata = clean_data["metadata"]
        
        print(f"Updating ChromaDB with {len(terminology)} terms...")
        existing_count = self.collection.count()
        
        # Prepare batch data
        chinese_terms = []
        metadatas = []
        ids = []
        
        current_time = datetime.now().isoformat()
        
        for chinese_term, data in terminology.items():
            chinese_terms.append(chinese_term)
            ids.append(self.generate_term_id(chinese_term))
            
            # Metadata for both initial and incremental
            metadatas.append({
                "english_term": data["english_term"],
                "category": data["category"],
                "frequency": data["frequency"],
                "confidence": data["confidence"],
                "chapters_seen": data["chapters_seen"],
                "first_seen": data["first_seen"],
                "last_seen": data["last_seen"],
                "created_at": data.get("created_at", current_time),
                "updated_at": current_time
            })
            
            # Progress logging every 50 terms
            if len(chinese_terms) % 50 == 0:
                print(f"Prepared {len(chinese_terms)}/{len(terminology)} terms...")
        
        # Single efficient batch upsert
        print("Embedding and storing terms in ChromaDB...")
        self.collection.upsert(
            documents=chinese_terms,
            metadatas=metadatas,
            ids=ids
        )
        
        new_count = self.collection.count()
        net_new_terms = new_count - existing_count
        
        print("ChromaDB update complete:")
        print(f"   - Terms processed: {len(terminology)}")
        print(f"   - Terms before: {existing_count}")
        print(f"   - Terms after: {new_count}")
        print(f"   - Net new terms: {net_new_terms}")
        print(f"   - Updated existing: {len(terminology) - net_new_terms}")
        
        return {
            "database_path": self.db_path,
            "collection_name": self.collection.name,
            "total_terms": new_count,
            "processed_terms": len(terminology),
            "net_new_terms": net_new_terms,
            "updated_terms": len(terminology) - net_new_terms,
            "categories": metadata["categories"],
            "embedding_model": self.embedding_model_name
        }
    
    def build_initial_vector_database(self, clean_data: Dict):
        terminology = clean_data["terminology"]
        metadata = clean_data["metadata"]
        
        print(f"Building initial vector database with {len(terminology)} terms...")
        
        # Prepare data for ChromaDB with consistent IDs
        chinese_terms = []
        metadatas = []
        ids = []
        
        for chinese_term, data in terminology.items():
            chinese_terms.append(chinese_term)
            
            metadatas.append({
                "english_term": data["english_term"],
                "category": data["category"],
                "frequency": data["frequency"],
                "confidence": data["confidence"],
                "chapters_seen": data["chapters_seen"],
                "first_seen": data["first_seen"],
                "last_seen": data["last_seen"],
                "created_at": data["created_at"]
            })
            
            # Use consistent hash-based IDs instead of sequential
            ids.append(self.generate_term_id(chinese_term))
        
        # Add to ChromaDB
        print("Embedding Chinese terms and storing in ChromaDB...")
        self.collection.upsert(
            documents=chinese_terms,
            metadatas=metadatas,
            ids=ids
        )
        
        print("Initial vector database built successfully")
        print(f"   - Total terms: {len(chinese_terms)}")
        print(f"   - Embeddings: {len(chinese_terms)} Chinese terms")
        print(f"   - Metadata: English translations + categories")
        print(f"   - Auto-saved to: {self.db_path}")
        
        return {
            "database_path": self.db_path,
            "collection_name": self.collection.name,
            "total_terms": len(chinese_terms),
            "categories": metadata["categories"],
            "embedding_model": metadata["embedding_model"]
        }
    
    def build_or_update_vector_database(self, clean_data: Dict):
        existing_count = self.collection.count()
        
        if existing_count == 0:
            print("No existing ChromaDB found - building initial database")
        else:
            print(f"Existing ChromaDB found with {existing_count} terms - updating incrementally")
        
        return self.update_vector_database(clean_data)
    
    def save_readable_summary(self, clean_data: Dict, db_info: Dict):
        terminology = clean_data["terminology"]
        metadata = clean_data["metadata"]
        
        with open(self.readable_file, 'w', encoding='utf-8') as f:
            f.write("CHROMADB VECTOR TERMINOLOGY DATABASE (BATCH OPTIMIZED)\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Database path: {db_info['database_path']}\n")
            f.write(f"Collection name: {db_info['collection_name']}\n")
            f.write(f"Embedding model: {db_info['embedding_model']}\n")
            f.write(f"Total terms: {db_info['total_terms']}\n")
            
            # Show update info
            if 'processed_terms' in db_info:
                f.write(f"Terms processed this batch: {db_info['processed_terms']}\n")
                f.write(f"Net new terms this batch: {db_info['net_new_terms']}\n")
                f.write(f"Updated existing terms: {db_info['updated_terms']}\n")
            
            f.write(f"Chapters covered: {metadata['chapters_covered']}\n")
            f.write(f"Average frequency: {metadata['avg_frequency']:.1f}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Show categories
            if 'categories' in db_info:
                f.write("CATEGORIES:\n")
                for category, count in sorted(db_info['categories'].items()):
                    f.write(f"  {category}: {count} terms\n")
                f.write("\n")
            
            # Group by category for readable display
            by_category = defaultdict(list)
            for chinese_term, data in terminology.items():
                by_category[data["category"]].append((chinese_term, data))
            
            for category, terms in sorted(by_category.items()):
                f.write(f"\n{category.upper()} ({len(terms)} terms):\n")
                f.write("-" * 50 + "\n")
                
                # Sort by frequency
                terms.sort(key=lambda x: x[1]["frequency"], reverse=True)
                
                for chinese_term, data in terms:
                    english_term = data["english_term"]
                    frequency = data["frequency"]
                    chapters_str = data.get("chapters_seen", "[]")
                    
                    # Parse chapters_seen back to list for display
                    try:
                        chapters = eval(chapters_str) if chapters_str != "[]" else []
                        chapter_range = f"{min(chapters)}-{max(chapters)}" if chapters else "?"
                    except:
                        chapter_range = "?"
                    
                    f.write(f"{chinese_term:15} → {english_term:30} (freq: {frequency}, chs: {chapter_range})\n")
        
        print(f"Readable summary saved to: {self.readable_file}")
    
    def test_database_setup(self):
        print("\nTesting incremental ChromaDB setup...")
        
        # Check collection exists
        collections = self.client.list_collections()
        print(f"Available collections: {[c.name for c in collections]}")
        
        # Check collection contents
        count = self.collection.count()
        print(f"Total documents in collection: {count}")
        
        if count > 0:
            # Peek at first few items
            sample = self.collection.peek(limit=3)
            print("\nSample data:")
            for i, (doc, metadata) in enumerate(zip(sample['documents'], sample['metadatas'])):
                print(f"  {i+1}. {doc} → {metadata['english_term']} ({metadata['category']})")
        
        print("Database test complete")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Build/Update ChromaDB Vector Terminology Database with Batch Processing")
    parser.add_argument("--bge", action="store_true", help="Use BGE-M3 embeddings (default)")
    parser.add_argument("--qwen", action="store_true", help="Use Qwen3-8B embeddings instead of BGE-M3")
    parser.add_argument("--lite", action="store_true", help="Use basic MPNet embeddings")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild from scratch (deletes existing)")
    
    args = parser.parse_args()
    
    print("Step 4: Building/Updating ChromaDB Vector Terminology Database (Batch Optimized)")
    print("=" * 75)
    
    # Configuration based on arguments
    if args.qwen:
        use_qwen3 = True
        use_bge = False
        print("Using Qwen3-8B embeddings")
    elif args.lite:
        use_qwen3 = False
        use_bge = False
        print("Using basic sentence-transformers embeddings")
    else:
        use_qwen3 = False
        use_bge = True
        print("Using BGE-M3 embeddings (default)")
    
    builder = ChromaTerminologyBuilder(
        use_qwen3=use_qwen3,
        use_bge=use_bge,
        qwen_model="Qwen/Qwen3-Embedding-8B"
    )
    
    # Handle rebuild flag
    if args.rebuild:
        print("--rebuild flag detected - deleting existing collection")
        try:
            builder.client.delete_collection(name=builder.collection_name)
            print("   Existing collection deleted")
            # Recreate collection
            builder.setup_chromadb()
        except Exception as e:
            print(f"   Note: {e} (collection may not have existed)")
    
    try:
        # Load extracted terminology
        print("Loading extracted terminology...")
        raw_data = builder.load_extracted_terminology()
        
        # Clean and prepare for vector database
        print("Cleaning and preparing terminology...")
        clean_data = builder.clean_and_prepare_terminology(raw_data)
        
        # Build or update ChromaDB vector database incrementally
        print("Building/updating ChromaDB vector database...")
        db_info = builder.build_or_update_vector_database(clean_data)
        
        # Save human-readable summary
        print("Saving readable summary...")
        builder.save_readable_summary(clean_data, db_info)
        
        # Test database
        builder.test_database_setup()
        
        print("\nIncremental ChromaDB Vector Database Update Complete")
        print("Vector database ready for step 5 (final translation)")
        print(f"Database location: {builder.db_path}")
        print(f"Embedding model: {builder.embedding_model_name}")
        print(f"Total terms now: {db_info['total_terms']}")
        print(f"This batch: +{db_info['net_new_terms']} new terms, {db_info['updated_terms']} updated")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run terminology extraction first:")
        print("  python 2b_extract_terminology.py --start 1 --end 3")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()