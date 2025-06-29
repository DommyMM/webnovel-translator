import os
import time
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
from openai import AsyncOpenAI
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

load_dotenv()

class NaiveTranslator:
    def __init__(self, chapter_num: int, start_chapter: int = 1):
        self.chapter_num = chapter_num
        self.start_chapter = start_chapter  # For progress bar positioning
        
        # File paths
        self.chinese_file = f"../data/chapters/clean/chapter_{chapter_num:04d}_cn.txt"
        self.output_file = f"../results/naive/translations/chapter_{chapter_num:04d}_naive.txt"
        
        # Setup directories
        Path("../results/naive/translations").mkdir(exist_ok=True, parents=True)
        
        # API client
        self.client = AsyncOpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
    
    def load_chinese_text(self) -> str:
        if not Path(self.chinese_file).exists():
            raise FileNotFoundError(f"Chinese file not found: {self.chinese_file}")
        
        with open(self.chinese_file, 'r', encoding='utf-8') as f:
            chinese_text = f.read().strip()
        
        return chinese_text
    
    async def naive_translate(self, chinese_text: str) -> str:
        # Minimal prompt - just translate as prose
        prompt = f"Translate this Chinese text to English prose:\n\n{chinese_text}"
        
        try:
            # Estimate tokens for progress bar
            estimated_total_tokens = int(len(chinese_text) * 1.31)
            
            # Initialize progress bar for this chapter
            with tqdm(
                total=estimated_total_tokens,
                desc=f"Ch {self.chapter_num} (naive)",
                unit="tok",
                unit_scale=True,
                position=self.chapter_num - self.start_chapter,
                leave=True,
                colour='yellow',
                smoothing=0.1,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            ) as pbar:
                
                # Enable streaming
                response = await self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=1.0,  # Default temperature
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
                        
                        # Better token estimation: ~4 chars per token for English output
                        new_tokens = len(new_content) / 4.0
                        tokens_received += new_tokens
                        
                        # Update progress bar
                        pbar.update(new_tokens)
                
                # Ensure progress bar reaches 100%
                if tokens_received < estimated_total_tokens:
                    pbar.update(estimated_total_tokens - tokens_received)
            
            return translation.strip()
            
        except Exception as e:
            print(f"Error in naive translation: {e}")
            return f"Translation failed: {e}"
    
    async def process_chapter_async(self, semaphore: asyncio.Semaphore) -> Optional[Dict]:
        async with semaphore:
            start_time = time.time()
            
            try:
                print(f"Starting Chapter {self.chapter_num} naive translation...")
                
                # Load Chinese text
                chinese_text = self.load_chinese_text()
                print(f"Chapter {self.chapter_num}: Loaded Chinese text - {len(chinese_text)} chars")
                
                # Do naive translation
                translation = await self.naive_translate(chinese_text)
                
                # Save translation
                with open(self.output_file, 'w', encoding='utf-8') as f:
                    f.write(f"Naive Translation - Chapter {self.chapter_num}\n")
                    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Prompt: 'Translate this Chinese text to English prose:'\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(translation)
                
                elapsed = time.time() - start_time
                
                result = {
                    "chapter": self.chapter_num,
                    "success": True,
                    "output_file": self.output_file,
                    "chinese_length": len(chinese_text),
                    "translation_length": len(translation),
                    "elapsed_time": elapsed
                }
                
                print(f"Chapter {self.chapter_num}: Naive translation complete ({elapsed:.1f}s)")
                print(f"  Output length: {len(translation)} chars")
                print(f"  Saved to: {self.output_file}")
                
                return result
                
            except Exception as e:
                print(f"Error processing Chapter {self.chapter_num}: {e}")
                return {
                    "chapter": self.chapter_num,
                    "success": False,
                    "error": str(e)
                }

async def run_naive_translation_pipeline(start_chapter: int, end_chapter: int, max_concurrent: int = 3):
    print("Step 6a: Naive Translation Pipeline")
    print("=" * 50)
    print(f"Generating naive translations for chapters {start_chapter}-{end_chapter}")
    print(f"Prompt: 'Translate this Chinese text to English prose:'")
    print(f"Max concurrent: {max_concurrent}")
    print()
    
    # Create semaphore for concurrency
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Create translators
    translators = []
    for chapter_num in range(start_chapter, end_chapter + 1):
        translator = NaiveTranslator(chapter_num, start_chapter)
        translators.append(translator)
    
    # Run translations concurrently
    print(f"Processing {len(translators)} chapters with streaming progress...")
    
    start_time = time.time()
    results = await asyncio.gather(*[t.process_chapter_async(semaphore) for t in translators])
    total_time = time.time() - start_time
    
    # Process results
    successful = [r for r in results if r and r.get("success")]
    failed = [r for r in results if r and not r.get("success")]
    
    print(f"\nStep 6a Complete")
    print(f"Total time: {total_time:.1f}s")
    print(f"Successful: {len(successful)}/{len(results)}")
    
    if successful:
        avg_time = sum(r["elapsed_time"] for r in successful) / len(successful)
        total_chars = sum(r["translation_length"] for r in successful)
        print(f"Average time per chapter: {avg_time:.1f}s")
        print(f"Total output: {total_chars:,} characters")
    
    if failed:
        print(f"Failed chapters: {len(failed)}")
        for result in failed:
            print(f"  Chapter {result['chapter']}: {result.get('error', 'Unknown error')}")
    
    # Save summary
    summary_file = "../results/naive/naive_translation_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("STEP 6a: NAIVE TRANSLATION SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Chapters processed: {start_chapter}-{end_chapter}\n")
        f.write(f"Total chapters: {len(results)}\n")
        f.write(f"Successful: {len(successful)}\n")
        f.write(f"Failed: {len(failed)}\n")
        f.write(f"Total time: {total_time:.1f}s\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Prompt used: 'Translate this Chinese text to English prose:'\n\n")
        
        if successful:
            f.write("SUCCESSFUL CHAPTERS:\n")
            f.write("-" * 30 + "\n")
            for result in successful:
                f.write(f"Chapter {result['chapter']:2d}: {result['elapsed_time']:5.1f}s, {result['translation_length']:,} chars\n")
        
        if failed:
            f.write(f"\nFAILED CHAPTERS:\n")
            f.write("-" * 20 + "\n")
            for result in failed:
                f.write(f"Chapter {result['chapter']:2d}: {result.get('error', 'Unknown error')}\n")
    
    print(f"\nOutput directory: ../results/naive/translations/")
    print(f"Summary saved to: {summary_file}")
    print("\nNext: Run step 6b to compare naive vs enhanced translations")

def main():
    parser = argparse.ArgumentParser(description="Step 6a: Generate naive translations with minimal prompt")
    parser.add_argument("--start", type=int, default=1, help="Start chapter number")
    parser.add_argument("--end", type=int, default=3, help="End chapter number")
    parser.add_argument("--concurrent", type=int, default=3, help="Max concurrent requests")
    
    args = parser.parse_args()
    
    # Check API key
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("Error: DEEPSEEK_API_KEY not found in environment")
        return
    
    # Check if Chinese chapters exist
    missing_files = []
    for chapter in range(args.start, args.end + 1):
        chinese_file = f"../data/chapters/clean/chapter_{chapter:04d}_cn.txt"
        if not Path(chinese_file).exists():
            missing_files.append(chinese_file)
    
    if missing_files:
        print("Error: Chinese chapter files not found:")
        for file in missing_files[:5]:
            print(f"  - {file}")
        if len(missing_files) > 5:
            print(f"  ... and {len(missing_files) - 5} more")
        return
    
    print("Step 6a Configuration:")
    print(f"  Chapters: {args.start}-{args.end}")
    print(f"  Max concurrent: {args.concurrent}")
    print(f"  Model: deepseek-chat")
    print(f"  Temperature: 1.0 (default)")
    print(f"  Prompt: 'Translate this Chinese text to English prose:'")
    print(f"  Output: ../results/naive/translations/")
    print()
    
    # Run naive translation pipeline
    asyncio.run(run_naive_translation_pipeline(args.start, args.end, args.concurrent))

if __name__ == "__main__":
    main()