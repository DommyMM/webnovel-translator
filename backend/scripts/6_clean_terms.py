import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from collections import defaultdict, Counter
from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv

load_dotenv()

class TerminologyDatabaseBuilder:
    """Build production-ready RAG database from extracted terminology"""
    
    def __init__(self):
        self.input_file = "../data/terminology/extracted_terminology.json"
        self.output_file = "../data/terminology/rag_database.json"
        self.readable_file = "../data/terminology/rag_database_readable.txt"
        self.setup_directories()
    
    def setup_directories(self):
        Path("../data/terminology").mkdir(exist_ok=True, parents=True)
    
    def load_extracted_terminology(self) -> Dict:
        """Load the raw extracted terminology"""
        if not Path(self.input_file).exists():
            raise FileNotFoundError(f"Extracted terminology not found: {self.input_file}")
        
        with open(self.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded extracted terminology with {data['metadata']['total_terms']} terms")
        return data
    
    def clean_and_merge_terminology(self, raw_data: Dict) -> Dict:
        """Clean and merge terminology entries"""
        
        raw_terminology = raw_data.get("terminology", {})
        
        # Build clean database
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
            
            # Skip empty or invalid entries
            if not chinese_term or not professional_term:
                continue
            
            # Skip single character terms (usually not important)
            if len(chinese_term) < 2:
                continue
            
            # Build clean entry
            clean_entry = {
                "english_term": professional_term,
                "category": category,
                "frequency": frequency,
                "confidence": confidence,
                "chapters_seen": sorted(chapters_seen) if chapters_seen else [],
                "first_seen": min(chapters_seen) if chapters_seen else 1,
                "last_seen": max(chapters_seen) if chapters_seen else 1,
                "created_at": datetime.now().isoformat()
            }
            
            clean_db[chinese_term] = clean_entry
            category_counts[category] += 1
            frequency_stats.append(frequency)
        
        # Build metadata
        metadata = {
            "created_at": datetime.now().isoformat(),
            "total_terms": len(clean_db),
            "source_method": "ai_extraction",
            "categories": dict(category_counts),
            "avg_frequency": sum(frequency_stats) / len(frequency_stats) if frequency_stats else 0,
            "max_frequency": max(frequency_stats) if frequency_stats else 0,
            "chapters_covered": raw_data["metadata"].get("chapters_processed", "unknown")
        }
        
        return {
            "metadata": metadata,
            "terminology": clean_db
        }
    
    def validate_key_terms(self, clean_db: Dict) -> List[str]:
        """Validate that key expected terms are present"""
        
        key_expected_terms = [
            "龙尘",  # Main character
            "丹帝",  # Title (the key fix)
            "灵根",  # Cultivation concept
            "宝儿"   # Supporting character
        ]
        
        missing_terms = []
        found_terms = []
        
        terminology = clean_db["terminology"]
        
        for term in key_expected_terms:
            if term in terminology:
                found_terms.append(f"{term} → {terminology[term]['english_term']}")
            else:
                missing_terms.append(term)
        
        print(f"\nKEY TERMS VALIDATION:")
        print("=" * 40)
        for term in found_terms:
            print(f"✅ {term}")
        
        if missing_terms:
            print(f"❌ Missing key terms: {missing_terms}")
        
        return missing_terms
    
    def save_rag_database(self, clean_db: Dict):
        """Save the production RAG database"""
        
        # Save JSON database
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(clean_db, f, indent=2, ensure_ascii=False)
        
        # Save human-readable version
        terminology = clean_db["terminology"]
        metadata = clean_db["metadata"]
        
        with open(self.readable_file, 'w', encoding='utf-8') as f:
            f.write("PRODUCTION RAG TERMINOLOGY DATABASE\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total terms: {metadata['total_terms']}\n")
            f.write(f"Chapters covered: {metadata['chapters_covered']}\n")
            f.write(f"Average frequency: {metadata['avg_frequency']:.1f}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Group by category
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
                    chapters = data.get("chapters_seen", [])
                    chapter_range = f"{min(chapters)}-{max(chapters)}" if chapters else "?"
                    
                    f.write(f"{chinese_term:15} → {english_term:30} (freq: {frequency}, chs: {chapter_range})\n")
        
        print(f"RAG database saved to: {self.output_file}")
        print(f"Readable version saved to: {self.readable_file}")
    
    def show_database_preview(self, clean_db: Dict):
        """Show preview of the RAG database"""
        
        terminology = clean_db["terminology"]
        metadata = clean_db["metadata"]
        
        print(f"\nRAG DATABASE PREVIEW:")
        print("=" * 50)
        print(f"Total terms: {metadata['total_terms']}")
        print(f"Categories: {metadata['categories']}")
        
        # Show top 10 most frequent terms
        terms_by_freq = sorted(
            terminology.items(),
            key=lambda x: x[1]["frequency"],
            reverse=True
        )
        
        print(f"\nTOP 10 MOST FREQUENT TERMS:")
        print("-" * 40)
        for i, (chinese_term, data) in enumerate(terms_by_freq[:10]):
            english_term = data["english_term"]
            frequency = data["frequency"]
            category = data["category"]
            print(f"{i+1:2d}. {chinese_term:12} → {english_term:20} ({category}, freq: {frequency})")
        
        # Show key terminology fix
        if "丹帝" in terminology:
            pill_god_data = terminology["丹帝"]
            print(f"\n🎯 KEY FIX CONFIRMED:")
            print(f"   丹帝 → {pill_god_data['english_term']} (frequency: {pill_god_data['frequency']})")
            print(f"   This will fix 'Alchemy Emperor' → 'Pill God' consistently!")

def test_rag_query(rag_db_file: str):
    """Test the RAG database with sample queries"""
    
    if not Path(rag_db_file).exists():
        print("RAG database not found for testing")
        return
    
    # Simple test query
    with open(rag_db_file, 'r', encoding='utf-8') as f:
        rag_data = json.load(f)
    
    terminology = rag_data["terminology"]
    
    # Test sample chapter terms
    test_terms = ["龙尘", "丹帝", "灵根", "宝儿", "虎骨丹"]
    
    print(f"\nRAG QUERY TEST:")
    print("=" * 40)
    print("Sample chapter terms: " + ", ".join(test_terms))
    print("\nRAG lookup results:")
    
    found_terms = {}
    for term in test_terms:
        if term in terminology:
            found_terms[term] = terminology[term]["english_term"]
            print(f"  {term} → {terminology[term]['english_term']}")
        else:
            print(f"  {term} → (not found)")
    
    print(f"\nTerminology injection for translation:")
    print("TERMINOLOGY (use these exact translations):")
    for cn, en in found_terms.items():
        print(f"  {cn} → {en}")
    
    return found_terms

def main():
    """Build production RAG database from extracted terminology"""
    
    print("Step 6: Building Production RAG Database")
    print("=" * 60)
    
    builder = TerminologyDatabaseBuilder()
    
    try:
        # Load extracted terminology
        raw_data = builder.load_extracted_terminology()
        
        # Clean and merge
        print("Cleaning and merging terminology...")
        clean_db = builder.clean_and_merge_terminology(raw_data)
        
        # Validate key terms
        missing_terms = builder.validate_key_terms(clean_db)
        
        # Show preview
        builder.show_database_preview(clean_db)
        
        # Save database
        print("\nSaving RAG database...")
        builder.save_rag_database(clean_db)
        
        # Test the database
        print("\nTesting RAG queries...")
        test_rag_query(builder.output_file)
        
        print(f"\n✅ RAG Database Build Complete!")
        print(f"Database ready for use in translation pipeline")
        
        if missing_terms:
            print(f"\n⚠️  Missing some expected terms: {missing_terms}")
            print("You may want to run terminology extraction on more chapters")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please run terminology extraction first:")
        print("  python 5_extract_terminology.py --start 1 --end 3")

if __name__ == "__main__":
    main()