import os
import time
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

@dataclass
class ChapterMetrics:
    chapter_num: int
    chinese_chars: int
    english_chars: int
    translation_time: float
    tokens_used: int
    tokens_per_second: float
    basic_similarity: float
    processing_time: float
    timestamp: str

@dataclass
class EnhancedConfig:
    chinese_chapters_dir: str = "clean_chapters"
    english_chapters_dir: str = "translated_chapters" 
    baseline_results_dir: str = "deepseek_results"    # For comparison
    output_dir: str = "enhanced_results"              # New enhanced results
    rules_file: str = "rules_clean.json"
    start_chapter: int = 1
    end_chapter: int = 3
    model: str = "deepseek-chat"
    temperature: float = 1.3
    max_tokens: int = 8192
    base_url: str = "https://api.deepseek.com"

class EnhancedTranslationPipeline:    # Enhanced translation pipeline that applies learned rules
    def __init__(self, config: EnhancedConfig):
        self.config = config
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=self.config.base_url
        )
        self.setup_directories()
        self.metrics: List[ChapterMetrics] = []
        self.rules = self.load_translation_rules()
        
    def setup_directories(self):
        Path(self.config.output_dir).mkdir(exist_ok=True)
        for subdir in ["translations", "comparisons", "analytics"]:
            Path(self.config.output_dir, subdir).mkdir(exist_ok=True)
    
    def load_translation_rules(self) -> List[Dict]:     # Load cleaned rules from rules_clean.json
        rules_file = Path(self.config.rules_file)
        if not rules_file.exists():
            print(f"Warning: Rules file not found: {rules_file}")
            return []
        
        with open(rules_file, 'r', encoding='utf-8') as f:
            rules_data = json.load(f)
        
        rules = rules_data.get("rules", [])
        print(f"Loaded {len(rules)} translation rules:")
        for i, rule in enumerate(rules, 1):
            print(f"  {i}. {rule.get('rule_type', 'unknown').upper()}: {rule.get('description', 'N/A')}")
        
        return rules
    
    def create_rules_prompt_section(self) -> str:       # Create the rules section for the translation prompt
        if not self.rules:
            return ""
        
        rules_text = "\nIMPORTANT TRANSLATION RULES:\n"
        rules_text += "Apply these specific rules when translating:\n\n"
        
        for i, rule in enumerate(self.rules, 1):
            rule_type = rule.get('rule_type', 'general').upper()
            description = rule.get('description', '')
            rules_text += f"{i}. {rule_type}: {description}\n"
        
        rules_text += "\nPrioritize following these rules while maintaining natural English flow.\n"
        return rules_text
    
    def load_chapter_files(self, chapter_num: int) -> tuple[str, str]:
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
    
    def create_enhanced_translation_prompt(self, chinese_text: str) -> str:     # Create translation prompt with learned rules injected
        rules_section = self.create_rules_prompt_section()
        
        prompt = f"""You are an expert translator specializing in Chinese web novels, particularly cultivation/xianxia genre. 

Your task is to translate the following Chinese chapter to English following specific learned rules for optimal quality.
{rules_section}
Guidelines:
- Keep character names in pinyin (e.g., Long Chen, not Dragon Chen)
- Translate cultivation realms consistently (e.g., 金丹期 → Golden Core stage)
- Preserve the action-oriented, dramatic tone typical of cultivation novels
- Use natural, engaging English prose that flows well

Chinese text to translate:

{chinese_text}

Please provide a high-quality English translation following the rules above:"""
        
        return prompt
    
    def translate_chapter(self, chinese_text: str) -> tuple[str, Dict]:
        start_time = time.time()
        prompt = self.create_enhanced_translation_prompt(chinese_text)
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "You are an expert translator specializing in Chinese cultivation novels. Follow the provided translation rules carefully."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            translation = response.choices[0].message.content
            
            # Clean up reasoning tags or code blocks
            translation = re.sub(r'<think>.*?</think>', '', translation, flags=re.DOTALL).strip()
            translation = re.sub(r'^```.*?```$', '', translation, flags=re.DOTALL | re.MULTILINE).strip()
            
            translation_time = time.time() - start_time
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
        trans_words = set(translation.lower().split())
        truth_words = set(ground_truth.lower().split())
        
        if not truth_words:
            return 0.0
            
        intersection = len(trans_words & truth_words)
        union = len(trans_words | truth_words)
        
        similarity = intersection / union if union > 0 else 0.0
        return min(similarity * 1.5, 1.0)
    
    def load_baseline_similarity(self, chapter_num: int) -> float:     # Load baseline similarity from previous run for comparison
        baseline_file = Path(self.config.baseline_results_dir, "comparisons", f"chapter_{chapter_num:04d}_comparison.json")
        if baseline_file.exists():
            try:
                with open(baseline_file, 'r', encoding='utf-8') as f:
                    baseline_data = json.load(f)
                return baseline_data.get("basic_similarity", 0.0)
            except Exception as e:
                print(f"Could not load baseline similarity for chapter {chapter_num}: {e}")
        return 0.0
    
    def save_chapter_results(self, chapter_num: int, chinese_text: str, translation: str, 
                           ground_truth: str, performance_stats: Dict, baseline_similarity: float):
        # Save enhanced translation
        translation_file = Path(self.config.output_dir, "translations", f"chapter_{chapter_num:04d}_enhanced.txt")
        with open(translation_file, 'w', encoding='utf-8') as f:
            f.write(f"Enhanced Translation with Rules - Chapter {chapter_num}\n")
            f.write(f"Rules Applied: {len(self.rules)}\n\n")
            f.write(translation)
        
        # Calculate current similarity
        enhanced_similarity = self.calculate_basic_similarity(translation, ground_truth)
        improvement = enhanced_similarity - baseline_similarity
        
        # Save comparison data
        comparison_file = Path(self.config.output_dir, "comparisons", f"chapter_{chapter_num:04d}_comparison.json")
        comparison_data = {
            "chapter": chapter_num,
            "model": self.config.model,
            "temperature": self.config.temperature,
            "rules_applied": len(self.rules),
            "chinese_length": len(chinese_text),
            "translation_length": len(translation),
            "ground_truth_length": len(ground_truth),
            "performance_stats": performance_stats,
            "enhanced_similarity": enhanced_similarity,
            "baseline_similarity": baseline_similarity,
            "improvement": improvement,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, indent=2, ensure_ascii=False)
        
        return enhanced_similarity, improvement
    
    def process_chapter(self, chapter_num: int) -> ChapterMetrics:
        print(f"\nProcessing Chapter {chapter_num} with Enhanced Rules")
        print("-" * 60)
        
        start_time = time.time()
        
        # Load input files
        try:
            chinese_text, ground_truth = self.load_chapter_files(chapter_num)
            print(f"Loaded files - Chinese: {len(chinese_text)} chars, Ground truth: {len(ground_truth)} chars")
        except FileNotFoundError as e:
            print(f"Error loading chapter {chapter_num}: {e}")
            return None
        
        # Load baseline similarity for comparison
        baseline_similarity = self.load_baseline_similarity(chapter_num)
        print(f"Baseline similarity: {baseline_similarity:.3f}")
        
        # Translate with enhanced rules
        print(f"Translating with {self.config.model} + {len(self.rules)} rules (temp={self.config.temperature})")
        translation, performance_stats = self.translate_chapter(chinese_text)
        
        if not translation:
            print(f"Translation failed for chapter {chapter_num}")
            return None
        
        # Save results and calculate improvement
        enhanced_similarity, improvement = self.save_chapter_results(
            chapter_num, chinese_text, translation, ground_truth, 
            performance_stats, baseline_similarity
        )
        
        # Create metrics record
        total_time = time.time() - start_time
        metrics = ChapterMetrics(
            chapter_num=chapter_num,
            chinese_chars=len(chinese_text),
            english_chars=len(translation),
            translation_time=performance_stats["translation_time"],
            tokens_used=performance_stats["total_tokens"],
            tokens_per_second=performance_stats["tokens_per_second"],
            basic_similarity=enhanced_similarity,
            processing_time=total_time,
            timestamp=datetime.now().isoformat()
        )
        
        self.metrics.append(metrics)
        
        # Show results
        print(f"Chapter {chapter_num} complete")
        print(f"Translation time: {performance_stats['translation_time']:.1f}s")
        print(f"Tokens: {performance_stats['total_tokens']} ({performance_stats['tokens_per_second']:.1f} tok/s)")
        print(f"Enhanced similarity: {enhanced_similarity:.3f}")
        print(f"Improvement: {improvement:+.3f}")
        print(f"Total time: {total_time:.1f}s")
        
        return metrics
    
    def save_final_analytics(self):
        analytics_file = Path(self.config.output_dir, "analytics", "enhanced_analytics.json")
        
        if not self.metrics:
            print("No metrics to save")
            return
        
        # Calculate summary statistics
        avg_similarity = sum(m.basic_similarity for m in self.metrics) / len(self.metrics)
        avg_tokens_per_sec = sum(m.tokens_per_second for m in self.metrics) / len(self.metrics)
        total_tokens = sum(m.tokens_used for m in self.metrics)
        total_translation_time = sum(m.translation_time for m in self.metrics)
        
        # Calculate overall improvement vs baseline
        improvements = []
        for chapter_num in range(self.config.start_chapter, self.config.end_chapter + 1):
            comparison_file = Path(self.config.output_dir, "comparisons", f"chapter_{chapter_num:04d}_comparison.json")
            if comparison_file.exists():
                with open(comparison_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    improvements.append(data.get("improvement", 0.0))
        
        avg_improvement = sum(improvements) / len(improvements) if improvements else 0.0
        
        analytics = {
            "config": asdict(self.config),
            "rules_applied": self.rules,
            "summary": {
                "chapters_processed": len(self.metrics),
                "model_used": self.config.model,
                "temperature": self.config.temperature,
                "rules_count": len(self.rules),
                "avg_enhanced_similarity": round(avg_similarity, 3),
                "avg_improvement": round(avg_improvement, 3),
                "avg_tokens_per_second": round(avg_tokens_per_sec, 2),
                "total_tokens_used": total_tokens,
                "total_translation_time": round(total_translation_time, 1),
                "avg_translation_time_per_chapter": round(total_translation_time / len(self.metrics), 1)
            },
            "chapter_metrics": [asdict(m) for m in self.metrics],
            "improvements": improvements,
            "generated_at": datetime.now().isoformat()
        }
        
        with open(analytics_file, 'w', encoding='utf-8') as f:
            json.dump(analytics, f, indent=2, ensure_ascii=False)
        
        print(f"Analytics saved to: {analytics_file}")
    
    def run_pipeline(self):
        print("Starting Enhanced Translation Pipeline with Rules")
        print(f"Model: {self.config.model}")
        print(f"Temperature: {self.config.temperature}")
        print(f"Rules loaded: {len(self.rules)}")
        print(f"Processing chapters {self.config.start_chapter}-{self.config.end_chapter}")
        
        start_time = time.time()
        
        for chapter_num in range(self.config.start_chapter, self.config.end_chapter + 1):
            try:
                metrics = self.process_chapter(chapter_num)
                if metrics:
                    print(f"Progress: {len(self.metrics)}/{self.config.end_chapter - self.config.start_chapter + 1} chapters")
            except Exception as e:
                print(f"Failed to process chapter {chapter_num}: {e}")
                continue
        
        # Final summary
        total_time = time.time() - start_time
        print(f"\nEnhanced Translation Pipeline Complete")
        print(f"Total time: {total_time:.1f}s")
        print(f"Chapters processed: {len(self.metrics)}")
        
        if self.metrics:
            avg_similarity = sum(m.basic_similarity for m in self.metrics) / len(self.metrics)
            avg_speed = sum(m.tokens_per_second for m in self.metrics) / len(self.metrics)
            print(f"Average enhanced similarity: {avg_similarity:.3f}")
            print(f"Average speed: {avg_speed:.1f} tokens/sec")
            
            # Show improvement summary
            improvements = []
            for chapter_num in range(self.config.start_chapter, self.config.end_chapter + 1):
                comparison_file = Path(self.config.output_dir, "comparisons", f"chapter_{chapter_num:04d}_comparison.json")
                if comparison_file.exists():
                    with open(comparison_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        improvements.append(data.get("improvement", 0.0))
            
            if improvements:
                avg_improvement = sum(improvements) / len(improvements)
                print(f"Average improvement over baseline: {avg_improvement:+.3f}")
        
        self.save_final_analytics()

def main():
    config = EnhancedConfig(
        start_chapter=1,
        end_chapter=3,
        model="deepseek-chat",
        temperature=1.3,
        max_tokens=8192
    )
    
    # Check API key
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("Error: DEEPSEEK_API_KEY not found in environment")
        return
    
    # Check if rules file exists
    if not Path(config.rules_file).exists():
        print(f"Error: Rules file not found: {config.rules_file}")
        print("Please run the rule cleaning pipeline first")
        return
    
    # Check if chapter directories exist
    if not Path(config.chinese_chapters_dir).exists():
        print(f"Error: Chinese chapters directory not found: {config.chinese_chapters_dir}")
        return
    
    if not Path(config.english_chapters_dir).exists():
        print(f"Error: English chapters directory not found: {config.english_chapters_dir}")
        return
    
    print("Enhanced Translation Configuration:")
    print(f"  Model: {config.model}")
    print(f"  Temperature: {config.temperature}")
    print(f"  Max tokens: {config.max_tokens}")
    print(f"  Chapters: {config.start_chapter}-{config.end_chapter}")
    print(f"  Rules file: {config.rules_file}")
    print(f"  Baseline results: {config.baseline_results_dir}")
    print(f"  Enhanced output: {config.output_dir}")
    
    # Initialize and run pipeline
    pipeline = EnhancedTranslationPipeline(config)
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()