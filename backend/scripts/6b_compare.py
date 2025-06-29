import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

class NaiveEnhancedComparator:
    def __init__(self, chapter_num: int):
        self.chapter_num = chapter_num
        
        # File paths
        self.chinese_file = f"../data/chapters/clean/chapter_{chapter_num:04d}_cn.txt"
        self.naive_file = f"../results/naive/translations/chapter_{chapter_num:04d}_naive.txt"
        self.enhanced_file = f"../results/final/translations/chapter_{chapter_num:04d}_final.txt"
        self.comparison_file = f"../results/comparison/chapter_{chapter_num:04d}_naive_vs_enhanced.txt"
        
        # Setup directories
        Path("../results/comparison").mkdir(exist_ok=True, parents=True)
    
    def load_files(self) -> Tuple[str, str, str]:
        # Load Chinese text
        if not Path(self.chinese_file).exists():
            raise FileNotFoundError(f"Chinese file not found: {self.chinese_file}")
        
        with open(self.chinese_file, 'r', encoding='utf-8') as f:
            chinese_text = f.read().strip()
        
        # Load naive translation from step 6a
        if not Path(self.naive_file).exists():
            raise FileNotFoundError(f"Naive translation not found: {self.naive_file}. Run step 6a first.")
        
        with open(self.naive_file, 'r', encoding='utf-8') as f:
            naive_content = f.read().strip()
            # Remove header (everything up to the separator line)
            if "=" * 60 in naive_content:
                naive_text = naive_content.split("=" * 60, 1)[1].strip()
            else:
                naive_text = naive_content
        
        # Load enhanced translation from step 5
        if not Path(self.enhanced_file).exists():
            raise FileNotFoundError(f"Enhanced translation not found: {self.enhanced_file}. Run step 5 first.")
        
        with open(self.enhanced_file, 'r', encoding='utf-8') as f:
            enhanced_text = f.read().strip()
        
        return chinese_text, naive_text, enhanced_text
    
    def calculate_basic_metrics(self, naive_text: str, enhanced_text: str) -> Dict:
        
        naive_words = naive_text.split()
        enhanced_words = enhanced_text.split()
        
        # Word overlap (Jaccard similarity)
        naive_word_set = set(word.lower() for word in naive_words)
        enhanced_word_set = set(word.lower() for word in enhanced_words)
        
        intersection = len(naive_word_set & enhanced_word_set)
        union = len(naive_word_set | enhanced_word_set)
        word_overlap = intersection / union if union > 0 else 0.0
        
        # Basic stats
        metrics = {
            "naive_char_count": len(naive_text),
            "enhanced_char_count": len(enhanced_text),
            "naive_word_count": len(naive_words),
            "enhanced_word_count": len(enhanced_words),
            "length_ratio": len(enhanced_text) / len(naive_text) if len(naive_text) > 0 else 0,
            "word_overlap": word_overlap,
            "char_difference": len(enhanced_text) - len(naive_text),
            "word_difference": len(enhanced_words) - len(naive_words)
        }
        
        return metrics
    
    def find_terminology_differences(self, naive_text: str, enhanced_text: str) -> List[str]:
        # Simple keyword spotting for obvious terminology differences
        
        # Common cultivation terms to look for
        cultivation_keywords = [
            "pill", "alchemy", "cultivation", "spiritual", "qi", "meridian",
            "breakthrough", "realm", "stage", "core", "foundation", "heaven",
            "emperor", "god", "master", "sect", "clan", "technique", "art"
        ]
        
        differences = []
        
        naive_lower = naive_text.lower()
        enhanced_lower = enhanced_text.lower()
        
        for keyword in cultivation_keywords:
            naive_count = naive_lower.count(keyword)
            enhanced_count = enhanced_lower.count(keyword)
            
            if naive_count != enhanced_count:
                differences.append(f"{keyword}: naive={naive_count}, enhanced={enhanced_count}")
        
        return differences[:10]  # Limit to top 10 differences
    
    def extract_key_phrases(self, text: str, max_phrases: int = 5) -> List[str]:
        # Extract some key phrases for comparison
        
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        key_phrases = []
        
        for sentence in sentences[:max_phrases]:
            sentence = sentence.strip()
            if len(sentence) > 20 and len(sentence) < 150:  # Reasonable length
                key_phrases.append(sentence)
        
        return key_phrases
    
    def create_side_by_side_comparison(self, chinese_text: str, naive_text: str, enhanced_text: str, metrics: Dict, terminology_diffs: List[str]):
        
        with open(self.comparison_file, 'w', encoding='utf-8') as f:
            f.write(f"NAIVE vs ENHANCED TRANSLATION COMPARISON - Chapter {self.chapter_num}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Naive source: Step 6a (minimal prompt)\n")
            f.write(f"Enhanced source: Step 5 (rules + RAG)\n\n")
            
            # Basic metrics
            f.write("BASIC METRICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Chinese length: {len(chinese_text):,} characters\n")
            f.write(f"Naive length: {metrics['naive_char_count']:,} characters ({metrics['naive_word_count']:,} words)\n")
            f.write(f"Enhanced length: {metrics['enhanced_char_count']:,} characters ({metrics['enhanced_word_count']:,} words)\n")
            f.write(f"Length ratio (enhanced/naive): {metrics['length_ratio']:.2f}\n")
            f.write(f"Character difference: {metrics['char_difference']:+,}\n")
            f.write(f"Word difference: {metrics['word_difference']:+,}\n")
            f.write(f"Word overlap (Jaccard): {metrics['word_overlap']:.3f}\n\n")
            
            # Terminology differences
            if terminology_diffs:
                f.write("TERMINOLOGY DIFFERENCES:\n")
                f.write("-" * 40 + "\n")
                for diff in terminology_diffs:
                    f.write(f"  {diff}\n")
                f.write("\n")
            
            # Chinese original (first 500 chars for reference)
            f.write("CHINESE ORIGINAL (first 500 chars):\n")
            f.write("-" * 40 + "\n")
            f.write(chinese_text[:500] + ("..." if len(chinese_text) > 500 else ""))
            f.write("\n\n")
            
            # Side by side comparison (first 1000 chars each)
            f.write("TRANSLATION COMPARISON (first 1000 chars each):\n")
            f.write("-" * 40 + "\n\n")
            
            f.write("NAIVE TRANSLATION (step 6a - minimal prompt):\n")
            f.write("~" * 40 + "\n")
            f.write(naive_text[:1000] + ("..." if len(naive_text) > 1000 else ""))
            f.write("\n\n")
            
            f.write("ENHANCED TRANSLATION (step 5 - rules + RAG):\n")
            f.write("~" * 40 + "\n")
            f.write(enhanced_text[:1000] + ("..." if len(enhanced_text) > 1000 else ""))
            f.write("\n\n")
            
            # Key phrases comparison
            naive_phrases = self.extract_key_phrases(naive_text)
            enhanced_phrases = self.extract_key_phrases(enhanced_text)
            
            f.write("KEY PHRASES COMPARISON:\n")
            f.write("-" * 40 + "\n")
            f.write("First few sentences from each:\n\n")
            
            for i in range(max(len(naive_phrases), len(enhanced_phrases))):
                f.write(f"Sentence {i+1}:\n")
                if i < len(naive_phrases):
                    f.write(f"  Naive: {naive_phrases[i]}\n")
                else:
                    f.write(f"  Naive: [no sentence {i+1}]\n")
                
                if i < len(enhanced_phrases):
                    f.write(f"  Enhanced: {enhanced_phrases[i]}\n")
                else:
                    f.write(f"  Enhanced: [no sentence {i+1}]\n")
                f.write("\n")
    
    def compare_chapter(self) -> Dict:
        print(f"Comparing Chapter {self.chapter_num}: Naive vs Enhanced...")
        
        try:
            # Load all files
            chinese_text, naive_text, enhanced_text = self.load_files()
            print(f"Chapter {self.chapter_num}: Loaded all files")
            
            # Calculate metrics
            metrics = self.calculate_basic_metrics(naive_text, enhanced_text)
            
            # Find terminology differences
            terminology_diffs = self.find_terminology_differences(naive_text, enhanced_text)
            
            # Create comparison file
            self.create_side_by_side_comparison(chinese_text, naive_text, enhanced_text, metrics, terminology_diffs)
            
            result = {
                "chapter": self.chapter_num,
                "success": True,
                "comparison_file": self.comparison_file,
                "metrics": metrics,
                "terminology_differences_count": len(terminology_diffs),
                "chinese_length": len(chinese_text)
            }
            
            print(f"Chapter {self.chapter_num}: Comparison complete")
            print(f"  Length ratio (enhanced/naive): {metrics['length_ratio']:.2f}")
            print(f"  Word overlap: {metrics['word_overlap']:.3f}")
            print(f"  Terminology differences: {len(terminology_diffs)}")
            print(f"  Saved to: {self.comparison_file}")
            
            return result
            
        except Exception as e:
            print(f"Error comparing Chapter {self.chapter_num}: {e}")
            return {
                "chapter": self.chapter_num,
                "success": False,
                "error": str(e)
            }

def run_comparison_pipeline(start_chapter: int, end_chapter: int):
    
    print("Step 6b: Naive vs Enhanced Comparison")
    print("=" * 50)
    print(f"Comparing chapters {start_chapter}-{end_chapter}")
    print("Naive (step 6a) vs Enhanced (step 5)")
    print()
    
    # Create comparators and run
    results = []
    for chapter_num in range(start_chapter, end_chapter + 1):
        comparator = NaiveEnhancedComparator(chapter_num)
        result = comparator.compare_chapter()
        results.append(result)
    
    # Process results
    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]
    
    print(f"\nStep 6b Complete")
    print(f"Successful: {len(successful)}/{len(results)}")
    
    if successful:
        # Calculate aggregate metrics
        avg_length_ratio = sum(r["metrics"]["length_ratio"] for r in successful) / len(successful)
        avg_word_overlap = sum(r["metrics"]["word_overlap"] for r in successful) / len(successful)
        total_terminology_diffs = sum(r["terminology_differences_count"] for r in successful)
        
        print(f"Average length ratio (enhanced/naive): {avg_length_ratio:.2f}")
        print(f"Average word overlap: {avg_word_overlap:.3f}")
        print(f"Total terminology differences found: {total_terminology_diffs}")
    
    if failed:
        print(f"Failed chapters: {len(failed)}")
        for result in failed:
            print(f"  Chapter {result['chapter']}: {result.get('error', 'Unknown error')}")
    
    # Save summary
    summary_file = "../results/comparison/comparison_summary.json"
    summary_data = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "chapters_processed": f"{start_chapter}-{end_chapter}",
            "total_chapters": len(results),
            "successful_chapters": len(successful),
            "failed_chapters": len(failed)
        },
        "aggregate_metrics": {
            "avg_length_ratio": avg_length_ratio if successful else 0,
            "avg_word_overlap": avg_word_overlap if successful else 0,
            "total_terminology_differences": total_terminology_diffs if successful else 0
        } if successful else {},
        "chapter_results": results
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    # Save human-readable summary
    readable_summary = "../results/comparison/comparison_summary.txt"
    with open(readable_summary, 'w', encoding='utf-8') as f:
        f.write("STEP 6b: NAIVE vs ENHANCED COMPARISON SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Chapters processed: {start_chapter}-{end_chapter}\n")
        f.write(f"Total chapters: {len(results)}\n")
        f.write(f"Successful: {len(successful)}\n")
        f.write(f"Failed: {len(failed)}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if successful:
            f.write("AGGREGATE METRICS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Average length ratio (enhanced/naive): {avg_length_ratio:.3f}\n")
            f.write(f"Average word overlap (Jaccard): {avg_word_overlap:.3f}\n")
            f.write(f"Total terminology differences: {total_terminology_diffs}\n\n")
            
            f.write("CHAPTER-BY-CHAPTER RESULTS:\n")
            f.write("-" * 40 + "\n")
            f.write("Ch# | Length Ratio | Word Overlap | Term Diffs\n")
            f.write("-" * 40 + "\n")
            for result in successful:
                ch = result["chapter"]
                lr = result["metrics"]["length_ratio"]
                wo = result["metrics"]["word_overlap"]
                td = result["terminology_differences_count"]
                f.write(f"{ch:2d}  |     {lr:5.2f}    |    {wo:5.3f}     |    {td:2d}\n")
        
        if failed:
            f.write(f"\nFAILED CHAPTERS:\n")
            f.write("-" * 20 + "\n")
            for result in failed:
                f.write(f"Chapter {result['chapter']}: {result.get('error', 'Unknown error')}\n")
    
    print(f"\nComparison files: ../results/comparison/")
    print(f"Summary: {readable_summary}")
    print(f"JSON data: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description="Step 6b: Compare naive vs enhanced translations")
    parser.add_argument("--start", type=int, default=1, help="Start chapter number")
    parser.add_argument("--end", type=int, default=3, help="End chapter number")
    
    args = parser.parse_args()
    
    # Check if required files exist
    missing_files = []
    for chapter in range(args.start, args.end + 1):
        naive_file = f"../results/naive/translations/chapter_{chapter:04d}_naive.txt"
        enhanced_file = f"../results/final/translations/chapter_{chapter:04d}_final.txt"
        
        if not Path(naive_file).exists():
            missing_files.append(f"Naive: {naive_file}")
        if not Path(enhanced_file).exists():
            missing_files.append(f"Enhanced: {enhanced_file}")
    
    if missing_files:
        print("Error: Required translation files not found:")
        for file in missing_files[:10]:
            print(f"  - {file}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")
        print("\nPlease run:")
        print("  - Step 6a (naive translation)")  
        print("  - Step 5 (enhanced translation)")
        return
    
    print("Step 6b Configuration:")
    print(f"  Chapters: {args.start}-{args.end}")
    print(f"  Naive source: ../results/naive/translations/ (step 6a)")
    print(f"  Enhanced source: ../results/final/translations/ (step 5)")
    print(f"  Output: ../results/comparison/")
    print()
    
    # Run comparison pipeline
    run_comparison_pipeline(args.start, args.end)

if __name__ == "__main__":
    main()