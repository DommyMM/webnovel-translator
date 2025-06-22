import os
import time
import json
import re
import argparse
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

@dataclass
class FinalTranslationMetrics:
    chapter_num: int
    chinese_chars: int
    english_chars: int
    translation_time: float
    tokens_used: int
    tokens_per_second: float
    basic_similarity: float
    processing_time: float
    rules_applied: int
    rag_terms_found: int
    terminology_applied: Dict[str, str]
    improvement_over_baseline: float
    improvement_over_enhanced: float
    timestamp: str

@dataclass
class FinalTranslationConfig:
    chinese_chapters_dir: str = "../data/chapters/clean"
    english_chapters_dir: str = "../data/chapters/ground_truth"
    baseline_results_dir: str = "../results/baseline"
    enhanced_results_dir: str = "../results/enhanced"
    output_dir: str = "../results/final"
    rules_file: str = "../data/rules/cleaned.json"
    rag_database_file: str = "../data/terminology/rag_database.json"
    start_chapter: int = 1
    end_chapter: int = 3
    model: str = "deepseek-chat"
    temperature: float = 1.3
    max_tokens: int = 8192
    base_url: str = "https://api.deepseek.com"
    max_concurrent: int = 10

class SimpleRAGQuerySystem:
    def __init__(self, rag_database_file: str):
        self.rag_database_file = rag_database_file
        self.terminology_db = None
        self.load_database()
    
    def load_database(self):
        if not Path(self.rag_database_file).exists():
            print(f"Warning: RAG database not found: {self.rag_database_file}")
            self.terminology_db = {}
            return
        
        with open(self.rag_database_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.terminology_db = data.get("terminology", {})
        print(f"Loaded RAG database with {len(self.terminology_db)} terms")
    
    def extract_chinese_terms(self, chinese_text: str) -> List[str]:
        terms = set()
        
        # Extract Chinese character sequences (2-6 characters)
        chinese_sequences = re.findall(r'[\u4e00-\u9fff]{2,6}', chinese_text)
        terms.update(chinese_sequences)
        
        # Filter out pure numbers and common words
        filtered_terms = []
        for term in terms:
            if (len(term) >= 2 and 
                re.match(r'^[\u4e00-\u9fff]+$', term) and
                not re.match(r'^[一二三四五六七八九十百千万]+$', term)):
                filtered_terms.append(term)
        
        return sorted(list(set(filtered_terms)))
    
    def query_terminology(self, chinese_terms: List[str]) -> Dict[str, str]:
        if not self.terminology_db:
            return {}
        
        results = {}
        for term in chinese_terms:
            if term in self.terminology_db:
                results[term] = self.terminology_db[term]["english_term"]
        
        return results
    
    def query_chapter(self, chinese_text: str) -> Dict[str, str]:
        chinese_terms = self.extract_chinese_terms(chinese_text)
        terminology = self.query_terminology(chinese_terms)
        return terminology

class AsyncFinalTranslator:
    def __init__(self, config: FinalTranslationConfig, chapter_num: int, rules: List[Dict], rag_system: SimpleRAGQuerySystem):
        self.config = config
        self.chapter_num = chapter_num
        self.rules = rules
        self.rag = rag_system
        self.client = AsyncOpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=self.config.base_url
        )
        self.setup_directories()
        
    def setup_directories(self):
        Path(self.config.output_dir).mkdir(exist_ok=True)
        for subdir in ["translations", "comparisons", "analytics"]:
            Path(self.config.output_dir, subdir).mkdir(exist_ok=True)
    
    def load_chapter_files(self) -> tuple[str, str]:
        cn_file = Path(self.config.chinese_chapters_dir) / f"chapter_{self.chapter_num:04d}_cn.txt"
        en_file = Path(self.config.english_chapters_dir) / f"chapter_{self.chapter_num:04d}_en.txt"
        
        if not cn_file.exists():
            raise FileNotFoundError(f"Chinese chapter file not found: {cn_file}")
        if not en_file.exists():
            raise FileNotFoundError(f"English chapter file not found: {en_file}")
            
        with open(cn_file, 'r', encoding='utf-8') as f:
            chinese_text = f.read().strip()
        with open(en_file, 'r', encoding='utf-8') as f:
            ground_truth = f.read().strip()
            
        return chinese_text, ground_truth
    
    def create_final_translation_prompt(self, chinese_text: str, terminology: Dict[str, str]) -> str:
        # Build rules section
        rules_section = ""
        if self.rules:
            rules_section = "\nTRANSLATION STYLE RULES:\n"
            rules_section += "Apply these specific style and structure rules:\n\n"
            
            for i, rule in enumerate(self.rules, 1):
                rule_type = rule.get('rule_type', 'general').upper()
                description = rule.get('description', '')
                rules_section += f"{i}. {rule_type}: {description}\n"
        
        # Build terminology section
        terminology_section = ""
        if terminology:
            terminology_section = "\nTERMINOLOGY (use these exact translations):\n"
            for chinese_term, english_term in terminology.items():
                terminology_section += f"  {chinese_term} → {english_term}\n"
            terminology_section += "\nIMPORTANT: Use these exact English terms when translating the corresponding Chinese terms.\n"
        
        # Combine into full prompt
        prompt = f"""You are an expert translator specializing in Chinese cultivation novels.

Your task is to translate the following Chinese chapter to English with perfect terminology consistency and style.
{rules_section}
{terminology_section}
Guidelines:
- Follow the style rules for natural flow and proper cultivation novel tone
- Use the provided terminology EXACTLY for consistent character names and cultivation terms
- Maintain the action-oriented, dramatic style typical of cultivation novels
- Use natural, engaging English prose that flows well

Chinese text to translate:

{chinese_text}

Please provide a high-quality English translation following the rules and terminology above:"""
        
        return prompt
    
    async def translate_chapter_async(self, chinese_text: str, terminology: Dict[str, str]) -> tuple[str, Dict]:
        start_time = time.time()
        prompt = self.create_final_translation_prompt(chinese_text, terminology)
        
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "You are an expert translator specializing in Chinese cultivation novels. Follow the provided translation rules and terminology carefully for perfect consistency."},
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
            print(f"Chapter {self.chapter_num}: Translation failed: {str(e)}")
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
    
    def load_previous_similarities(self) -> tuple[float, float]:
        baseline_similarity = 0.0
        enhanced_similarity = 0.0
        
        # Load baseline similarity
        baseline_file = Path(self.config.baseline_results_dir, "comparisons", f"chapter_{self.chapter_num:04d}_comparison.json")
        if baseline_file.exists():
            try:
                with open(baseline_file, 'r', encoding='utf-8') as f:
                    baseline_data = json.load(f)
                baseline_similarity = baseline_data.get("basic_similarity", 0.0)
            except Exception:
                pass
        
        # Load enhanced similarity
        enhanced_file = Path(self.config.enhanced_results_dir, "comparisons", f"chapter_{self.chapter_num:04d}_comparison.json")
        if enhanced_file.exists():
            try:
                with open(enhanced_file, 'r', encoding='utf-8') as f:
                    enhanced_data = json.load(f)
                enhanced_similarity = enhanced_data.get("enhanced_similarity", 0.0)
            except Exception:
                pass
        
        return baseline_similarity, enhanced_similarity
    
    def save_chapter_results(self, chinese_text: str, translation: str, ground_truth: str, 
                            performance_stats: Dict, baseline_similarity: float, enhanced_similarity: float,
                            terminology_applied: Dict[str, str]) -> tuple[float, float, float]:
        # Save final translation
        translation_file = Path(self.config.output_dir, "translations", f"chapter_{self.chapter_num:04d}_final.txt")
        with open(translation_file, 'w', encoding='utf-8') as f:
            f.write(f"Final Translation with Rules + RAG - Chapter {self.chapter_num}\n")
            f.write(f"Style Rules Applied: {len(self.rules)}\n")
            f.write(f"RAG Terms Applied: {len(terminology_applied)}\n")
            if terminology_applied:
                f.write("Terminology Used:\n")
                for cn, en in terminology_applied.items():
                    f.write(f"  {cn} → {en}\n")
            f.write("\n")
            f.write(translation)
        
        # Calculate similarities and improvements
        final_similarity = self.calculate_basic_similarity(translation, ground_truth)
        improvement_over_baseline = final_similarity - baseline_similarity
        improvement_over_enhanced = final_similarity - enhanced_similarity
        
        # Save comparison data
        comparison_file = Path(self.config.output_dir, "comparisons", f"chapter_{self.chapter_num:04d}_comparison.json")
        comparison_data = {
            "chapter": self.chapter_num,
            "model": self.config.model,
            "temperature": self.config.temperature,
            "rules_applied": len(self.rules),
            "rag_terms_applied": len(terminology_applied),
            "terminology_used": terminology_applied,
            "chinese_length": len(chinese_text),
            "translation_length": len(translation),
            "ground_truth_length": len(ground_truth),
            "performance_stats": performance_stats,
            "final_similarity": final_similarity,
            "baseline_similarity": baseline_similarity,
            "enhanced_similarity": enhanced_similarity,
            "improvement_over_baseline": improvement_over_baseline,
            "improvement_over_enhanced": improvement_over_enhanced,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, indent=2, ensure_ascii=False)
        
        return final_similarity, improvement_over_baseline, improvement_over_enhanced
    
    async def process_chapter_async(self, semaphore: asyncio.Semaphore) -> Optional[FinalTranslationMetrics]:
        async with semaphore:
            print(f"Starting Chapter {self.chapter_num} final translation (Rules + RAG)...")
            
            start_time = time.time()
            
            # Load input files
            try:
                chinese_text, ground_truth = self.load_chapter_files()
                print(f"Chapter {self.chapter_num}: Loaded files - Chinese: {len(chinese_text)} chars")
            except FileNotFoundError as e:
                print(f"Chapter {self.chapter_num}: Error loading: {e}")
                return None
            
            # Query RAG for relevant terminology
            terminology = self.rag.query_chapter(chinese_text)
            print(f"Chapter {self.chapter_num}: Found {len(terminology)} RAG mappings")
            
            if terminology:
                print(f"Chapter {self.chapter_num}: Using terminology: {list(terminology.items())[:5]}...")
            
            # Load previous similarities for comparison
            baseline_similarity, enhanced_similarity = self.load_previous_similarities()
            
            # Translate with full Rules + RAG
            print(f"Chapter {self.chapter_num}: Translating with {self.config.model} + {len(self.rules)} rules + {len(terminology)} RAG terms...")
            translation, performance_stats = await self.translate_chapter_async(chinese_text, terminology)
            
            if not translation:
                print(f"Chapter {self.chapter_num}: Translation failed")
                return None
            
            # Save results and calculate improvements
            final_similarity, improvement_over_baseline, improvement_over_enhanced = self.save_chapter_results(
                chinese_text, translation, ground_truth, 
                performance_stats, baseline_similarity, enhanced_similarity, terminology
            )
            
            # Create metrics record
            total_time = time.time() - start_time
            metrics = FinalTranslationMetrics(
                chapter_num=self.chapter_num,
                chinese_chars=len(chinese_text),
                english_chars=len(translation),
                translation_time=performance_stats["translation_time"],
                tokens_used=performance_stats["total_tokens"],
                tokens_per_second=performance_stats["tokens_per_second"],
                basic_similarity=final_similarity,
                processing_time=total_time,
                rules_applied=len(self.rules),
                rag_terms_found=len(terminology),
                terminology_applied=terminology,
                improvement_over_baseline=improvement_over_baseline,
                improvement_over_enhanced=improvement_over_enhanced,
                timestamp=datetime.now().isoformat()
            )
            
            # Show completion
            print(f"Chapter {self.chapter_num} complete: {performance_stats['translation_time']:.1f}s")
            print(f"  Final similarity: {final_similarity:.3f}")
            print(f"  vs Baseline: {improvement_over_baseline:+.3f}")
            print(f"  vs Enhanced: {improvement_over_enhanced:+.3f}")
            print(f"  RAG terms: {len(terminology)}")
            
            return metrics

class AsyncFinalTranslationPipeline:    
    def __init__(self, config: FinalTranslationConfig):
        self.config = config
        self.rules = self.load_translation_rules()
        self.rag = self.setup_rag_system()
        self.setup_directories()
        
    def setup_directories(self):
        Path(self.config.output_dir).mkdir(exist_ok=True)
        for subdir in ["translations", "comparisons", "analytics"]:
            Path(self.config.output_dir, subdir).mkdir(exist_ok=True)
    
    def load_translation_rules(self) -> List[Dict]:
        rules_file = Path(self.config.rules_file)
        if not rules_file.exists():
            print(f"Warning: Rules file not found: {rules_file}")
            return []
        
        with open(rules_file, 'r', encoding='utf-8') as f:
            rules_data = json.load(f)
        
        rules = rules_data.get("rules", [])
        print(f"Loaded {len(rules)} style rules")
        return rules
    
    def setup_rag_system(self) -> SimpleRAGQuerySystem:
        rag = SimpleRAGQuerySystem(self.config.rag_database_file)
        return rag
    
    async def run_async_final_translation(self):
        print("Starting Final Translation Pipeline with Rules + RAG")
        print(f"Model: {self.config.model}")
        print(f"Temperature: {self.config.temperature}")
        print(f"Style rules loaded: {len(self.rules)}")
        print(f"RAG database loaded: {len(self.rag.terminology_db) if self.rag.terminology_db else 0} terms")
        print(f"Max concurrent requests: {self.config.max_concurrent}")
        print(f"Processing chapters {self.config.start_chapter}-{self.config.end_chapter}")
        
        start_time = time.time()
        chapters = list(range(self.config.start_chapter, self.config.end_chapter + 1))
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        # Create translators for each chapter
        translators = [AsyncFinalTranslator(self.config, chapter_num, self.rules, self.rag) 
                        for chapter_num in chapters]
        
        # Create tasks for all chapters
        tasks = [translator.process_chapter_async(semaphore) for translator in translators]
        
        print(f"Launching {len(tasks)} concurrent final translation tasks...")
        
        # Run all tasks concurrently
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful_metrics = []
        failed_count = 0
        
        for i, result in enumerate(results_list):
            chapter_num = chapters[i]
            if isinstance(result, Exception):
                print(f"Chapter {chapter_num} failed with exception: {result}")
                failed_count += 1
            elif result is not None:
                successful_metrics.append(result)
            else:
                failed_count += 1
        
        translation_time = time.time() - start_time
        print(f"Final translation complete in {translation_time:.1f}s")
        print(f"Successfully processed {len(successful_metrics)}/{len(chapters)} chapters")
        print(f"Failed chapters: {failed_count}")
        
        # Save analytics
        print("Saving analytics...")
        self.save_final_analytics(successful_metrics)
        
        # Show results summary
        if successful_metrics:
            avg_final_similarity = sum(m.basic_similarity for m in successful_metrics) / len(successful_metrics)
            avg_improvement_baseline = sum(m.improvement_over_baseline for m in successful_metrics) / len(successful_metrics)
            avg_improvement_enhanced = sum(m.improvement_over_enhanced for m in successful_metrics) / len(successful_metrics)
            total_rag_terms = sum(m.rag_terms_found for m in successful_metrics)
            
            print(f"\nFINAL RESULTS SUMMARY:")
            print("=" * 50)
            print(f"Average final similarity: {avg_final_similarity:.3f}")
            print(f"Average improvement over baseline: {avg_improvement_baseline:+.3f}")
            print(f"Average improvement over enhanced: {avg_improvement_enhanced:+.3f}")
            print(f"Total RAG terms applied: {total_rag_terms}")
            
            # Show terminology examples
            all_terminology = {}
            for m in successful_metrics:
                all_terminology.update(m.terminology_applied)
            
            if all_terminology:
                print(f"\nKEY TERMINOLOGY APPLIED ({len(all_terminology)} unique terms):")
                for cn, en in list(all_terminology.items())[:8]:
                    print(f"  {cn} → {en}")
        
        return successful_metrics
    
    def save_final_analytics(self, metrics: List[FinalTranslationMetrics]):
        analytics_file = Path(self.config.output_dir, "analytics", "final_analytics.json")
        
        if not metrics:
            print("No metrics to save")
            return
        
        # Calculate comprehensive statistics
        avg_final_similarity = sum(m.basic_similarity for m in metrics) / len(metrics)
        avg_improvement_baseline = sum(m.improvement_over_baseline for m in metrics) / len(metrics)
        avg_improvement_enhanced = sum(m.improvement_over_enhanced for m in metrics) / len(metrics)
        
        # Collect all unique terminology used
        all_terminology = {}
        for m in metrics:
            all_terminology.update(m.terminology_applied)
        
        analytics = {
            "config": asdict(self.config),
            "rules_applied": self.rules,
            "summary": {
                "chapters_processed": len(metrics),
                "model_used": self.config.model,
                "temperature": self.config.temperature,
                "rules_count": len(self.rules),
                "rag_terms_available": len(self.rag.terminology_db) if self.rag.terminology_db else 0,
                "avg_final_similarity": round(avg_final_similarity, 3),
                "avg_improvement_over_baseline": round(avg_improvement_baseline, 3),
                "avg_improvement_over_enhanced": round(avg_improvement_enhanced, 3),
                "unique_terminology_used": len(all_terminology),
                "processing_method": "async_concurrent"
            },
            "terminology_database_used": all_terminology,
            "chapter_metrics": [asdict(m) for m in metrics],
            "generated_at": datetime.now().isoformat()
        }
        
        with open(analytics_file, 'w', encoding='utf-8') as f:
            json.dump(analytics, f, indent=2, ensure_ascii=False)
        
        print(f"Final analytics saved to: {analytics_file}")

def main():
    parser = argparse.ArgumentParser(description="Final Translation Pipeline with Rules + RAG")
    parser.add_argument("--chapter", type=int, help="Process single chapter")
    parser.add_argument("--start", type=int, default=1, help="Start chapter")
    parser.add_argument("--end", type=int, default=3, help="End chapter")
    parser.add_argument("--concurrent", type=int, default=3, help="Max concurrent requests")
    
    args = parser.parse_args()
    
    config = FinalTranslationConfig(
        start_chapter=args.start,
        end_chapter=args.end,
        max_concurrent=args.concurrent
    )
    
    # Check API key
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("Error: DEEPSEEK_API_KEY not found in environment")
        return
    
    # Check if required files exist
    required_files = [
        config.rules_file,
        config.rag_database_file
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("Error: Required files not found:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\nPlease run the prerequisite steps:")
        print("  1. python 3_clean_rules.py (for rules)")
        print("  2. python 5_extract_terminology.py (for terminology)")
        print("  3. python 6_clean_terminology.py (for RAG database)")
        return
    
    if args.chapter:
        # Single chapter mode
        print(f"Processing single chapter: {args.chapter}")
        async def single_chapter():
            pipeline = AsyncFinalTranslationPipeline(config)
            semaphore = asyncio.Semaphore(1)
            translator = AsyncFinalTranslator(config, args.chapter, pipeline.rules, pipeline.rag)
            result = await translator.process_chapter_async(semaphore)
            if result:
                print(f"Chapter {args.chapter} final translation complete")
                print(f"  Rules applied: {result.rules_applied}")
                print(f"  RAG terms: {result.rag_terms_found}")
                print(f"  Final similarity: {result.basic_similarity:.3f}")
            else:
                print(f"Chapter {args.chapter} processing failed")
        
        asyncio.run(single_chapter())
    else:
        # Process all chapters
        print("Final Translation Configuration:")
        print(f"  Model: {config.model}")
        print(f"  Temperature: {config.temperature}")
        print(f"  Max concurrent: {config.max_concurrent}")
        print(f"  Chapters: {config.start_chapter}-{config.end_chapter}")
        print(f"  Rules file: {config.rules_file}")
        print(f"  RAG database: {config.rag_database_file}")
        print(f"  Output: {config.output_dir}")
        
        pipeline = AsyncFinalTranslationPipeline(config)
        asyncio.run(pipeline.run_async_final_translation())

if __name__ == "__main__":
    main()