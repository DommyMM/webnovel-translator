import os
import time
import json
import re
import asyncio
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from openai import AsyncOpenAI
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

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
    retry_count: int
    timestamp: str

@dataclass
class PipelineConfig:
    chinese_chapters_dir: str = "../data/chapters/clean"
    english_chapters_dir: str = "../data/chapters/ground_truth" 
    output_dir: str = "../results/baseline"
    start_chapter: int = 1
    end_chapter: int = 3
    model: str = "deepseek-chat"
    temperature: float = 1.3
    max_tokens: int = 8192
    base_url: str = "https://api.deepseek.com"
    max_concurrent: int = 10
    max_retries: int = 3
    base_retry_delay: float = 2.0

class ParallelDeepSeekPipeline:    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.client = AsyncOpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=self.config.base_url
        )
        self.setup_directories()
        self.metrics: List[ChapterMetrics] = []
        
    def setup_directories(self):
        Path(self.config.output_dir).mkdir(exist_ok=True, parents=True)
        for subdir in ["translations", "comparisons", "analytics"]:
            Path(self.config.output_dir, subdir).mkdir(exist_ok=True, parents=True)
    
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
    
    def create_translation_prompt(self, chinese_text: str) -> str:
        """Create translation prompt optimized for DeepSeek reasoning"""
        prompt = """You are an expert translator specializing in Chinese web novels, particularly cultivation/xianxia genre. 

Your task is to translate the following Chinese chapter to English. Think through the translation systematically:

1. First understand the context and genre-specific terminology
2. Identify character names, cultivation terms, and cultural elements
3. Maintain narrative flow and readability in English
4. Ensure consistency with established xianxia translation conventions

Guidelines:
- Keep character names in pinyin (e.g., Long Chen, not Dragon Chen or Dragon Dust)
- Translate cultivation realms consistently (e.g., 金丹期 → Golden Core stage)
- Preserve the action-oriented, dramatic tone typical of cultivation novels
- Use natural, engaging English prose that flows well

Chinese text to translate:

{chinese_text}

Please provide a high-quality English translation:"""
        
        return prompt.format(chinese_text=chinese_text)
    
    async def translate_chapter_async(self, chinese_text: str, chapter_num: int) -> tuple[str, Dict]:
        prompt = self.create_translation_prompt(chinese_text)
        retry_count = 0
        
        for attempt in range(self.config.max_retries + 1):
            start_time = time.time()
            
            try:
                # Estimate tokens for progress bar using our actual API ratio
                estimated_total_tokens = int(len(chinese_text) * 1.31)
                
                # Initialize progress bar for this chapter (fix positioning for any start chapter)
                with tqdm(
                    total=estimated_total_tokens,
                    desc=f"Chapter {chapter_num}",
                    unit="tok",
                    unit_scale=True,
                    position=chapter_num - self.config.start_chapter,
                    leave=True,
                    colour='blue',
                    smoothing=0.1,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
                ) as pbar:
                    
                    # Enable streaming
                    response = await self.client.chat.completions.create(
                        model=self.config.model,
                        messages=[
                            {"role": "system", "content": "You are an expert translator specializing in Chinese cultivation novels."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
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
                
                # Clean up translation (same as before)
                translation = re.sub(r'<think>.*?</think>', '', translation, flags=re.DOTALL).strip()
                translation = re.sub(r'^```.*?```$', '', translation, flags=re.DOTALL | re.MULTILINE).strip()
                
                translation_time = time.time() - start_time
                
                # Mock token usage (since streaming doesn't return usage)
                total_tokens = max(tokens_received, estimated_total_tokens)
                tokens_per_second = total_tokens / translation_time if translation_time > 0 else 0
                
                performance_stats = {
                    "translation_time": translation_time,
                    "total_tokens": total_tokens,
                    "tokens_per_second": tokens_per_second,
                    "prompt_tokens": int(len(chinese_text) * 0.6),
                    "completion_tokens": int(len(translation.split()) * 1.31),
                    "retry_count": retry_count
                }
                
                if retry_count > 0:
                    print(f"Chapter {chapter_num}: Success after {retry_count} retries")
                
                return translation, performance_stats
                
            except Exception as e:
                retry_count += 1
                error_msg = str(e)
                
                # Check if this is a retryable error
                retryable_errors = [
                    "connection", "timeout", "network", "503", "502", "500", 
                    "rate limit", "temporarily unavailable", "internal server error"
                ]
                
                is_retryable = any(keyword.lower() in error_msg.lower() for keyword in retryable_errors)
                
                if attempt < self.config.max_retries and is_retryable:
                    # Exponential backoff with jitter
                    delay = self.config.base_retry_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"Chapter {chapter_num}: Retryable error on attempt {attempt + 1}/{self.config.max_retries + 1}: {error_msg}")
                    print(f"Chapter {chapter_num}: Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
                else:
                    print(f"Chapter {chapter_num}: Translation failed after {retry_count} retries: {error_msg}")
                    return "", {
                        "translation_time": 0, 
                        "total_tokens": 0, 
                        "tokens_per_second": 0,
                        "retry_count": retry_count,
                        "error": error_msg
                    }
        
        # Should never reach here, but just in case
        return "", {
            "translation_time": 0, 
            "total_tokens": 0, 
            "tokens_per_second": 0,
            "retry_count": retry_count,
            "error": "Max retries exceeded"
        }
    
    def calculate_basic_similarity(self, translation: str, ground_truth: str) -> float:
        """Calculate word overlap similarity"""
        trans_words = set(translation.lower().split())
        truth_words = set(ground_truth.lower().split())
        
        if not truth_words:
            return 0.0
            
        intersection = len(trans_words & truth_words)
        union = len(trans_words | truth_words)
        
        # Jaccard similarity with slight scaling for readability
        similarity = intersection / union if union > 0 else 0.0
        return min(similarity * 1.5, 1.0)
    
    def save_chapter_results(self, chapter_num: int, chinese_text: str, translation: str, ground_truth: str, performance_stats: Dict):
        # Save translation
        translation_file = Path(self.config.output_dir, "translations", f"chapter_{chapter_num:04d}_deepseek.txt")
        with open(translation_file, 'w', encoding='utf-8') as f:
            f.write(f"DeepSeek Translation - Chapter {chapter_num}\n\n")
            f.write(translation)
        
        # Save comparison data
        comparison_file = Path(self.config.output_dir, "comparisons", f"chapter_{chapter_num:04d}_comparison.json")
        comparison_data = {
            "chapter": chapter_num,
            "model": self.config.model,
            "temperature": self.config.temperature,
            "chinese_length": len(chinese_text),
            "translation_length": len(translation),
            "ground_truth_length": len(ground_truth),
            "performance_stats": performance_stats,
            "basic_similarity": self.calculate_basic_similarity(translation, ground_truth),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, indent=2, ensure_ascii=False)
    
    async def process_chapter_async(self, chapter_num: int, semaphore: asyncio.Semaphore) -> Optional[ChapterMetrics]:
        """Process a single chapter asynchronously with rate limiting"""
        async with semaphore:  # Limit concurrent requests
            print(f"Starting Chapter {chapter_num}...")
            
            start_time = time.time()
            
            # Load input files
            try:
                chinese_text, ground_truth = self.load_chapter_files(chapter_num)
                print(f"Chapter {chapter_num} loaded - Chinese: {len(chinese_text)} chars, Ground truth: {len(ground_truth)} chars")
            except FileNotFoundError as e:
                print(f"Error loading chapter {chapter_num}: {e}")
                return None
            
            # Translate with DeepSeek (async with retry)
            translation, performance_stats = await self.translate_chapter_async(chinese_text, chapter_num)
            
            if not translation:
                print(f"Translation failed for chapter {chapter_num}")
                return None
                
            # Calculate similarity metrics
            basic_similarity = self.calculate_basic_similarity(translation, ground_truth)
            
            # Save results
            self.save_chapter_results(chapter_num, chinese_text, translation, ground_truth, performance_stats)
            
            # Create metrics record
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
                retry_count=performance_stats.get("retry_count", 0),
                timestamp=datetime.now().isoformat()
            )
            
            # Show completion
            retry_info = f" (after {performance_stats.get('retry_count', 0)} retries)" if performance_stats.get('retry_count', 0) > 0 else ""
            print(f"Chapter {chapter_num} complete ({performance_stats['translation_time']:.1f}s, {performance_stats['total_tokens']} tokens, similarity: {basic_similarity:.3f}){retry_info}")
            
            return metrics
    
    def save_final_analytics(self):
        analytics_file = Path(self.config.output_dir, "analytics", "deepseek_analytics.json")
        
        if not self.metrics:
            print("No metrics to save")
            return
            
        # Calculate summary statistics
        avg_similarity = sum(m.basic_similarity for m in self.metrics) / len(self.metrics)
        avg_tokens_per_sec = sum(m.tokens_per_second for m in self.metrics) / len(self.metrics)
        total_tokens = sum(m.tokens_used for m in self.metrics)
        total_translation_time = sum(m.translation_time for m in self.metrics)
        total_retries = sum(m.retry_count for m in self.metrics)
        
        analytics = {
            "config": asdict(self.config),
            "summary": {
                "chapters_processed": len(self.metrics),
                "model_used": self.config.model,
                "temperature": self.config.temperature,
                "avg_similarity": round(avg_similarity, 3),
                "avg_tokens_per_second": round(avg_tokens_per_sec, 2),
                "total_tokens_used": total_tokens,
                "total_translation_time": round(total_translation_time, 1),
                "avg_translation_time_per_chapter": round(total_translation_time / len(self.metrics), 1),
                "total_retries": total_retries,
                "chapters_requiring_retries": sum(1 for m in self.metrics if m.retry_count > 0),
                "parallel_processing": True,
                "max_concurrent_requests": self.config.max_concurrent,
                "retry_configuration": {
                    "max_retries": self.config.max_retries,
                    "base_retry_delay": self.config.base_retry_delay
                }
            },
            "chapter_metrics": [asdict(m) for m in self.metrics],
            "generated_at": datetime.now().isoformat()
        }
        
        with open(analytics_file, 'w', encoding='utf-8') as f:
            json.dump(analytics, f, indent=2, ensure_ascii=False)
        
        print(f"Analytics saved to: {analytics_file}")
    
    async def run_pipeline_async(self):
        """Run the parallel translation pipeline"""
        print("Starting Parallel DeepSeek Translation Pipeline")
        print(f"Model: {self.config.model}")
        print(f"Temperature: {self.config.temperature}")
        print(f"Max concurrent requests: {self.config.max_concurrent}")
        print(f"Max retries per chapter: {self.config.max_retries}")
        print(f"Processing chapters {self.config.start_chapter}-{self.config.end_chapter}")
        
        start_time = time.time()
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        # Create tasks for all chapters
        tasks = []
        for chapter_num in range(self.config.start_chapter, self.config.end_chapter + 1):
            task = self.process_chapter_async(chapter_num, semaphore)
            tasks.append(task)
        
        # Run all tasks concurrently
        print(f"Launching {len(tasks)} parallel translation tasks...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful_metrics = []
        failed_count = 0
        
        for result in results:
            if isinstance(result, Exception):
                print(f"Task failed with exception: {result}")
                failed_count += 1
            elif result is not None:
                successful_metrics.append(result)
            else:
                failed_count += 1
        
        self.metrics = successful_metrics
        
        # Final summary
        total_time = time.time() - start_time
        print(f"\nParallel DeepSeek Translation Pipeline Complete")
        print(f"Total time: {total_time:.1f}s")
        print(f"Chapters processed: {len(successful_metrics)}")
        print(f"Failed chapters: {failed_count}")
        
        if successful_metrics:
            avg_similarity = sum(m.basic_similarity for m in successful_metrics) / len(successful_metrics)
            avg_speed = sum(m.tokens_per_second for m in successful_metrics) / len(successful_metrics)
            total_tokens = sum(m.tokens_used for m in successful_metrics)
            avg_translation_time = sum(m.translation_time for m in successful_metrics) / len(successful_metrics)
            total_retries = sum(m.retry_count for m in successful_metrics)
            chapters_with_retries = sum(1 for m in successful_metrics if m.retry_count > 0)
            
            print(f"Average similarity: {avg_similarity:.3f}")
            print(f"Average speed: {avg_speed:.1f} tokens/sec")
            print(f"Total tokens: {total_tokens}")
            print(f"Average translation time per chapter: {avg_translation_time:.1f}s")
            print(f"Total retries: {total_retries} ({chapters_with_retries} chapters required retries)")
        
        self.save_final_analytics()

def main():
    parser = argparse.ArgumentParser(description="Step 1: Baseline Translation Pipeline")
    parser.add_argument("--start", type=int, default=1, help="Start chapter number")
    parser.add_argument("--end", type=int, default=3, help="End chapter number")
    parser.add_argument("--concurrent", type=int, default=10, help="Max concurrent requests")
    
    args = parser.parse_args()
    
    config = PipelineConfig(
        start_chapter=args.start,
        end_chapter=args.end,
        model="deepseek-chat",
        temperature=1.3,
        max_tokens=8192,
        max_concurrent=args.concurrent,
        max_retries=3,
        base_retry_delay=2.0
    )
    
    # Check API key
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("Error: DEEPSEEK_API_KEY not found in environment")
        print("Please set your DeepSeek API key in .env file")
        return
    
    # Check if chapter directories exist
    if not Path(config.chinese_chapters_dir).exists():
        print(f"Error: Chinese chapters directory not found: {config.chinese_chapters_dir}")
        return
    
    if not Path(config.english_chapters_dir).exists():
        print(f"Error: English chapters directory not found: {config.english_chapters_dir}")
        return
    
    print("Configuration:")
    print(f"  Model: {config.model}")
    print(f"  Temperature: {config.temperature}")
    print(f"  Max tokens: {config.max_tokens}")
    print(f"  Max concurrent: {config.max_concurrent}")
    print(f"  Max retries: {config.max_retries}")
    print(f"  Base retry delay: {config.base_retry_delay}s")
    print(f"  Chapters: {config.start_chapter}-{config.end_chapter}")
    print(f"  Chinese dir: {config.chinese_chapters_dir}")
    print(f"  English dir: {config.english_chapters_dir}")
    print(f"  Output dir: {config.output_dir}")
    
    # Initialize and run pipeline
    pipeline = ParallelDeepSeekPipeline(config)
    asyncio.run(pipeline.run_pipeline_async())

if __name__ == "__main__":
    main()