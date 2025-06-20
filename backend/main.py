import os
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv

# Load environment variables from current directory (backend)
load_dotenv()

@dataclass
class ChapterMetrics:
    """Track metrics for each chapter processing"""
    chapter_num: int
    
    # Content metrics
    chinese_chars: int
    english_chars: int
    
    # Performance metrics  
    translation_time: float
    tokens_used: int
    tokens_per_second: float
    
    # Quality metrics (placeholder for now)
    basic_similarity: float
    
    # Processing info
    processing_time: float
    timestamp: str

@dataclass
class PipelineConfig:
    """Configuration for the Phase 1 pipeline"""
    # Paths (matching your directory structure)
    chinese_chapters_dir: str = "clean_chapters"
    english_chapters_dir: str = "translated_chapters" 
    output_dir: str = "phase1_results"
    
    # Processing
    start_chapter: int = 1
    end_chapter: int = 3  # Start small for testing
    
    # Model settings (using qwen-3-32b with 16k context)
    model: str = "qwen-3-32b"
    temperature: float = 0.1
    max_tokens: int = 4000
    max_context: int = 16382  # qwen-3-32b context limit

class Phase1Pipeline:
    """Main orchestrator for Phase 1 raw translation testing"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.client = Cerebras(api_key=os.getenv("CEREBRAS_API_KEY"))
        self.setup_directories()
        self.metrics: List[ChapterMetrics] = []
        
    def setup_directories(self):
        """Create necessary output directories"""
        Path(self.config.output_dir).mkdir(exist_ok=True)
        
        # Create subdirectories for organization
        for subdir in ["raw_translations", "comparisons", "analytics"]:
            Path(self.config.output_dir, subdir).mkdir(exist_ok=True)
    
    def load_chapter_files(self, chapter_num: int) -> tuple[str, str]:
        """Load Chinese and English files for a chapter"""
        cn_file = Path(self.config.chinese_chapters_dir) / f"chapter_{chapter_num:04d}_cn.txt"
        en_file = Path(self.config.english_chapters_dir) / f"chapter_{chapter_num:04d}_en.txt"
        
        if not cn_file.exists():
            raise FileNotFoundError(f"Chinese chapter file not found: {cn_file}")
        if not en_file.exists():
            raise FileNotFoundError(f"English chapter file not found: {en_file}")
            
        with open(cn_file, 'r', encoding='utf-8') as f:
            chinese_text = f.read().strip()
        with open(en_file, 'r', encoding='utf-8') as f:
            ground_truth = f.read().strip()
            
        return chinese_text, ground_truth
    
    def create_translation_prompt(self, chinese_text: str) -> str:
        """Create basic translation prompt for raw testing"""
        prompt = """You are an expert translator specializing in Chinese web novels, particularly cultivation/xianxia genre. 
Translate the following Chinese chapter to English with these priorities:

1. Maintain narrative flow and readability
2. Keep character names consistent, using pinyin only for these names
3. Preserve cultural context and technical terminology
4. Use natural, engaging English prose

Chinese text to translate:

{chinese_text}"""
        
        return prompt.format(chinese_text=chinese_text)
    
    def translate_chapter(self, chinese_text: str) -> tuple[str, Dict]:
        """Translate a chapter using Cerebras and track performance"""
        start_time = time.time()
        
        prompt = self.create_translation_prompt(chinese_text)
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            translation = response.choices[0].message.content
            translation_time = time.time() - start_time
            
            # Calculate performance metrics
            total_tokens = response.usage.total_tokens
            tokens_per_second = total_tokens / translation_time if translation_time > 0 else 0
            
            performance_stats = {
                "translation_time": translation_time,
                "total_tokens": total_tokens,
                "tokens_per_second": tokens_per_second,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens
            }
            
            return translation, performance_stats
            
        except Exception as e:
            print(f"Translation failed: {str(e)}")
            return "", {"translation_time": 0, "total_tokens": 0, "tokens_per_second": 0}
    
    def calculate_basic_similarity(self, translation: str, ground_truth: str) -> float:
        """Calculate basic similarity between translation and ground truth"""
        # Simple word overlap similarity for now
        trans_words = set(translation.lower().split())
        truth_words = set(ground_truth.lower().split())
        
        if not truth_words:
            return 0.0
            
        intersection = len(trans_words & truth_words)
        union = len(trans_words | truth_words)
        
        # Jaccard similarity 
        similarity = intersection / union if union > 0 else 0.0
        return similarity
    
    def save_chapter_results(self, chapter_num: int, chinese_text: str, translation: str, ground_truth: str, performance_stats: Dict):
        """Save chapter processing results"""
        # Save raw translation
        raw_file = Path(self.config.output_dir, "raw_translations", f"chapter_{chapter_num:04d}_raw.txt")
        with open(raw_file, 'w', encoding='utf-8') as f:
            f.write(translation)
        
        # Save comparison data
        comparison_file = Path(self.config.output_dir, "comparisons", f"chapter_{chapter_num:04d}_comparison.json")
        comparison_data = {
            "chapter": chapter_num,
            "chinese_length": len(chinese_text),
            "translation_length": len(translation),
            "ground_truth_length": len(ground_truth),
            "performance_stats": performance_stats,
            "basic_similarity": self.calculate_basic_similarity(translation, ground_truth),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, indent=2, ensure_ascii=False)
    
    def process_chapter(self, chapter_num: int) -> ChapterMetrics:
        """Process a single chapter - raw translation and basic metrics"""
        print(f"\nProcessing chapter {chapter_num}")
        
        start_time = time.time()
        
        # Step 1: Load files
        try:
            chinese_text, ground_truth = self.load_chapter_files(chapter_num)
        except FileNotFoundError as e:
            print(f"Error loading chapter {chapter_num}: {e}")
            return None
        
        # Step 2: Raw translation
        print(f"Translating with {self.config.model}")
        translation, performance_stats = self.translate_chapter(chinese_text)
        
        if not translation:
            print(f"Translation failed for chapter {chapter_num}")
            return None
            
        # Step 3: Calculate basic similarity
        basic_similarity = self.calculate_basic_similarity(translation, ground_truth)
        
        # Step 4: Save results
        self.save_chapter_results(chapter_num, chinese_text, translation, 
                                ground_truth, performance_stats)
        
        # Step 5: Create metrics
        total_time = time.time() - start_time
        metrics = ChapterMetrics(
            chapter_num=chapter_num,
            chinese_chars=len(chinese_text),
            english_chars=len(translation),
            translation_time=performance_stats["translation_time"],
            tokens_used=performance_stats["total_tokens"],
            tokens_per_second=performance_stats["tokens_per_second"],
            basic_similarity=basic_similarity,
            processing_time=total_time,
            timestamp=datetime.now().isoformat()
        )
        
        self.metrics.append(metrics)
        
        # Show results
        print(f"Chapter {chapter_num} complete")
        print(f"Translation time: {performance_stats['translation_time']:.1f}s")
        print(f"Tokens: {performance_stats['total_tokens']} ({performance_stats['tokens_per_second']:.1f} tok/s)")
        print(f"Basic similarity: {basic_similarity:.3f}")
        print(f"Total time: {total_time:.1f}s")
        
        return metrics
    
    def save_final_analytics(self):
        """Save comprehensive analytics for the entire run"""
        analytics_file = Path(self.config.output_dir, "analytics", "phase1_raw_analytics.json")
        
        if not self.metrics:
            print("No metrics to save")
            return
            
        # Calculate summary stats
        avg_similarity = sum(m.basic_similarity for m in self.metrics) / len(self.metrics)
        avg_tokens_per_sec = sum(m.tokens_per_second for m in self.metrics) / len(self.metrics)
        total_tokens = sum(m.tokens_used for m in self.metrics)
        total_translation_time = sum(m.translation_time for m in self.metrics)
        
        analytics = {
            "config": asdict(self.config),
            "summary": {
                "chapters_processed": len(self.metrics),
                "avg_basic_similarity": round(avg_similarity, 3),
                "avg_tokens_per_second": round(avg_tokens_per_sec, 2),
                "total_tokens_used": total_tokens,
                "total_translation_time": round(total_translation_time, 1),
                "avg_translation_time_per_chapter": round(total_translation_time / len(self.metrics), 1)
            },
            "chapter_metrics": [asdict(m) for m in self.metrics],
            "generated_at": datetime.now().isoformat()
        }
        
        with open(analytics_file, 'w', encoding='utf-8') as f:
            json.dump(analytics, f, indent=2, ensure_ascii=False)
        
        print(f"Analytics saved to: {analytics_file}")
    
    def run_pipeline(self):
        """Execute the Phase 1 raw translation pipeline"""
        print("Starting Phase 1 raw translation pipeline")
        print(f"Processing chapters {self.config.start_chapter}-{self.config.end_chapter}")
        print(f"Using model: {self.config.model}")
        
        start_time = time.time()
        
        for chapter_num in range(self.config.start_chapter, self.config.end_chapter + 1):
            self.process_chapter(chapter_num)
        
        # Final summary
        total_time = time.time() - start_time
        print(f"\nPhase 1 raw translation complete")
        print(f"Total time: {total_time:.1f}s")
        print(f"Chapters processed: {len(self.metrics)}")
        
        if self.metrics:
            avg_similarity = sum(m.basic_similarity for m in self.metrics) / len(self.metrics)
            print(f"Average similarity: {avg_similarity:.3f}")
        
        self.save_final_analytics()

def main():
    """Main entry point for Phase 1 raw translation testing"""
    # Configuration for raw translation testing
    config = PipelineConfig(
        start_chapter=1,
        end_chapter=3,  # Start with 3 chapters for testing
        model="qwen-3-32b",  # Use qwen with 16k context
        temperature=0.1,
        max_tokens=4000
    )
    
    # Check API key
    if not os.getenv("CEREBRAS_API_KEY"):
        print("Error: CEREBRAS_API_KEY not found in environment")
        return
    
    # Check if chapter directories exist
    if not Path(config.chinese_chapters_dir).exists():
        print(f"Error: Chinese chapters directory not found: {config.chinese_chapters_dir}")
        return
    
    if not Path(config.english_chapters_dir).exists():
        print(f"Error: English chapters directory not found: {config.english_chapters_dir}")
        return
    
    print("Configuration:")
    print(f"Model: {config.model}")
    print(f"Chapters: {config.start_chapter}-{config.end_chapter}")
    print(f"Chinese dir: {config.chinese_chapters_dir}")
    print(f"English dir: {config.english_chapters_dir}")
    print(f"Output dir: {config.output_dir}")
    
    # Initialize and run pipeline
    pipeline = Phase1Pipeline(config)
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()