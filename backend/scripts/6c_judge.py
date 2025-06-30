import os
import json
import asyncio
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

load_dotenv()

class LLMJudge:
    def __init__(self, chapter_num: int, start_chapter: int = 1):
        self.chapter_num = chapter_num
        self.start_chapter = start_chapter  # For progress bar positioning
        
        # File paths
        self.naive_file = f"../results/naive/translations/chapter_{chapter_num:04d}_naive.txt"
        self.enhanced_file = f"../results/final/translations/chapter_{chapter_num:04d}_final.txt"
        self.judgment_file = f"../results/evaluation/judgments/chapter_{chapter_num:04d}_judgment.json"
        
        # Setup directories
        Path("../results/evaluation/judgments").mkdir(exist_ok=True, parents=True)
        
        # API client - using Cerebras Qwen for evaluation
        self.client = Cerebras(api_key=os.getenv("CEREBRAS_API_KEY"))
    
    def load_translations(self) -> tuple[str, str]:
        """Load naive and enhanced translations"""
        
        # Load naive translation
        if not Path(self.naive_file).exists():
            raise FileNotFoundError(f"Naive translation not found: {self.naive_file}")
        
        with open(self.naive_file, 'r', encoding='utf-8') as f:
            naive_content = f.read().strip()
            # Remove header (everything up to the separator line)
            if "=" * 60 in naive_content:
                naive_text = naive_content.split("=" * 60, 1)[1].strip()
            else:
                naive_text = naive_content
        
        # Load enhanced translation
        if not Path(self.enhanced_file).exists():
            raise FileNotFoundError(f"Enhanced translation not found: {self.enhanced_file}")
        
        with open(self.enhanced_file, 'r', encoding='utf-8') as f:
            enhanced_text = f.read().strip()
        
        return naive_text, enhanced_text
    
    def create_judgment_prompt(self, naive_text: str, enhanced_text: str) -> str:
        """Create LLM judgment prompt for head-to-head comparison"""
        
        prompt = f"""You are evaluating two English translations of the same Chinese cultivation novel chapter. Pick which translation is better and explain why.

TRANSLATION A (Naive):
{naive_text}

TRANSLATION B (Enhanced):
{enhanced_text}

Your task: Compare these translations as a Western reader who enjoys cultivation novels.

Consider:
- Reading Flow: Which reads more smoothly and naturally?
- Terminology: Which uses better cultivation novel terminology?
- Character Voice: Which makes characters feel more real?
- Story Clarity: Which is easier to follow and more engaging?
- Overall Quality: Which would you prefer to read?

Response Format:
WINNER: [A or B]
CONFIDENCE: [High/Medium/Low]
REASON: [2-3 sentence explanation of why the winner is better]

Example:
WINNER: B
CONFIDENCE: High
REASON: Translation B flows more naturally and uses proper cultivation terminology like "Pill God" instead of "Alchemy Emperor". The sentence structure feels more like natural English prose rather than a literal translation.

Your judgment:"""
        
        return prompt
    
    async def judge_translations(self, naive_text: str, enhanced_text: str) -> Dict:
        prompt = self.create_judgment_prompt(naive_text, enhanced_text)
        
        try:
            # Estimate tokens for progress (much smaller than translation)
            estimated_tokens = 200  # Just for judgment, much smaller
            
            with tqdm(
                total=estimated_tokens,
                desc=f"Ch {self.chapter_num} (judge)",
                unit="tok",
                unit_scale=True,
                position=self.chapter_num - self.start_chapter,
                leave=True,
                colour='purple',
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]"
            ) as pbar:
                
                # Cerebras synchronous call (run in thread to avoid blocking)
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.client.chat.completions.create(
                        model="qwen-3-32b",  # Cerebras Qwen model
                        messages=[
                            {"role": "system", "content": "You are an expert judge of English translation quality for cultivation novels. Be decisive and give clear preferences."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.1,  # Low temp for consistent judgments
                        max_tokens=500,   # Short response
                    )
                )
                
                judgment_text = response.choices[0].message.content
                pbar.update(estimated_tokens)  # Complete progress bar
            
            return self.parse_judgment(judgment_text)
            
        except Exception as e:
            print(f"Error getting LLM judgment: {e}")
            return {
                "winner": "unknown",
                "confidence": "low", 
                "reason": f"Error: {e}",
                "raw_response": ""
            }
    
    def parse_judgment(self, judgment_text: str) -> Dict:
        judgment = {
            "winner": "unknown",
            "confidence": "low",
            "reason": "Could not parse response",
            "raw_response": judgment_text
        }
        
        lines = judgment_text.split('\n')
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("WINNER:"):
                winner_text = line.replace("WINNER:", "").strip().upper()
                if 'A' in winner_text:
                    judgment["winner"] = "naive"
                elif 'B' in winner_text:
                    judgment["winner"] = "enhanced"
            
            elif line.startswith("CONFIDENCE:"):
                conf_text = line.replace("CONFIDENCE:", "").strip().lower()
                if conf_text in ["high", "medium", "low"]:
                    judgment["confidence"] = conf_text
            
            elif line.startswith("REASON:"):
                reason_text = line.replace("REASON:", "").strip()
                if len(reason_text) > 10:  # Valid reason
                    judgment["reason"] = reason_text
        
        return judgment
    
    async def judge_chapter_async(self, semaphore: asyncio.Semaphore) -> Optional[Dict]:
        async with semaphore:
            start_time = time.time()
            
            try:
                print(f"Starting Chapter {self.chapter_num} LLM judgment...")
                
                # Load both translations
                naive_text, enhanced_text = self.load_translations()
                print(f"Chapter {self.chapter_num}: Loaded translations - Naive: {len(naive_text)} chars, Enhanced: {len(enhanced_text)} chars")
                
                # Get LLM judgment
                judgment = await self.judge_translations(naive_text, enhanced_text)
                
                # Create result
                result = {
                    "chapter": self.chapter_num,
                    "success": True,
                    "judgment": judgment,
                    "metadata": {
                        "naive_length": len(naive_text),
                        "enhanced_length": len(enhanced_text),
                        "length_ratio": len(enhanced_text) / len(naive_text) if len(naive_text) > 0 else 0,
                        "judgment_time": time.time() - start_time,
                        "timestamp": datetime.now().isoformat()
                    }
                }
                
                # Save individual judgment
                with open(self.judgment_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                elapsed = time.time() - start_time
                winner_text = {"naive": "Naive", "enhanced": "Enhanced", "unknown": "Unclear"}[judgment["winner"]]
                
                print(f"Chapter {self.chapter_num}: Judgment complete ({elapsed:.1f}s)")
                print(f"  Winner: {winner_text} ({judgment['confidence']} confidence)")
                print(f"  Reason: {judgment['reason'][:80]}...")
                print(f"  Saved to: {self.judgment_file}")
                
                return result
                
            except Exception as e:
                print(f"Error judging Chapter {self.chapter_num}: {e}")
                return {
                    "chapter": self.chapter_num,
                    "success": False,
                    "error": str(e)
                }

async def run_llm_judgment_pipeline(start_chapter: int, end_chapter: int, max_concurrent: int = 3):
    print("Step 6c: LLM Head-to-Head Judgment")
    print("=" * 50)
    print(f"Judging chapters {start_chapter}-{end_chapter}")
    print("Model: qwen-3-32b (Cerebras)")
    print("Task: Pick which translation is better (Naive vs Enhanced)")
    print(f"Max concurrent: {max_concurrent}")
    print()
    
    # Create semaphore for concurrency
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Create judges
    judges = []
    for chapter_num in range(start_chapter, end_chapter + 1):
        judge = LLMJudge(chapter_num, start_chapter)
        judges.append(judge)
    
    # Run judgments concurrently
    print(f"Processing {len(judges)} chapters with LLM judgment...")
    
    start_time = time.time()
    results = await asyncio.gather(*[j.judge_chapter_async(semaphore) for j in judges])
    total_time = time.time() - start_time
    
    # Process results
    successful = [r for r in results if r and r.get("success")]
    failed = [r for r in results if r and not r.get("success")]
    
    print(f"\nStep 6c Complete")
    print(f"Total time: {total_time:.1f}s")
    print(f"Successful: {len(successful)}/{len(results)}")
    
    # Analyze judgments
    if successful:
        naive_wins = sum(1 for r in successful if r["judgment"]["winner"] == "naive")
        enhanced_wins = sum(1 for r in successful if r["judgment"]["winner"] == "enhanced")
        unclear = sum(1 for r in successful if r["judgment"]["winner"] == "unknown")
        
        high_conf = sum(1 for r in successful if r["judgment"]["confidence"] == "high")
        medium_conf = sum(1 for r in successful if r["judgment"]["confidence"] == "medium")
        low_conf = sum(1 for r in successful if r["judgment"]["confidence"] == "low")
        
        print(f"\nJUDGMENT RESULTS:")
        print(f"Enhanced wins: {enhanced_wins}/{len(successful)} ({enhanced_wins/len(successful)*100:.1f}%)")
        print(f"Naive wins: {naive_wins}/{len(successful)} ({naive_wins/len(successful)*100:.1f}%)")
        print(f"Unclear: {unclear}/{len(successful)}")
        print(f"\nCONFIDENCE LEVELS:")
        print(f"High confidence: {high_conf}/{len(successful)}")
        print(f"Medium confidence: {medium_conf}/{len(successful)}")
        print(f"Low confidence: {low_conf}/{len(successful)}")
    
    if failed:
        print(f"\nFailed chapters: {len(failed)}")
        for result in failed:
            print(f"  Chapter {result['chapter']}: {result.get('error', 'Unknown error')}")
    
    # Save aggregate results
    save_aggregate_results(successful, failed, start_chapter, end_chapter, total_time)
    
    print(f"\nOutput directory: ../results/evaluation/")
    print(f"Individual judgments: ../results/evaluation/judgments/")
    print(f"Summary: ../results/evaluation/llm_judgment_summary.txt")

def save_aggregate_results(successful: list, failed: list, start_chapter: int, end_chapter: int, total_time: float):
    # Calculate statistics
    if successful:
        naive_wins = [r for r in successful if r["judgment"]["winner"] == "naive"]
        enhanced_wins = [r for r in successful if r["judgment"]["winner"] == "enhanced"]
        unclear = [r for r in successful if r["judgment"]["winner"] == "unknown"]
        
        high_conf = [r for r in successful if r["judgment"]["confidence"] == "high"]
        medium_conf = [r for r in successful if r["judgment"]["confidence"] == "medium"]
        low_conf = [r for r in successful if r["judgment"]["confidence"] == "low"]
        
        # Calculate average judgment time
        avg_judgment_time = sum(r["metadata"]["judgment_time"] for r in successful) / len(successful)
    else:
        naive_wins = enhanced_wins = unclear = high_conf = medium_conf = low_conf = []
        avg_judgment_time = 0
    
    # Save JSON summary
    json_summary = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "chapters_processed": f"{start_chapter}-{end_chapter}",
            "total_chapters": len(successful) + len(failed),
            "successful_chapters": len(successful),
            "failed_chapters": len(failed),
            "total_time": total_time,
            "avg_judgment_time": avg_judgment_time,
            "model_used": "qwen-3-32b"
        },
        "judgment_summary": {
            "enhanced_wins": len(enhanced_wins),
            "naive_wins": len(naive_wins),
            "unclear": len(unclear),
            "enhanced_win_rate": len(enhanced_wins) / len(successful) if successful else 0,
            "high_confidence": len(high_conf),
            "medium_confidence": len(medium_conf),
            "low_confidence": len(low_conf)
        },
        "chapter_results": successful + failed
    }
    
    json_file = "../results/evaluation/llm_judgment_summary.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_summary, f, indent=2, ensure_ascii=False)
    
    # Save readable summary
    readable_file = "../results/evaluation/llm_judgment_summary.txt"
    with open(readable_file, 'w', encoding='utf-8') as f:
        f.write("STEP 6c: LLM HEAD-TO-HEAD JUDGMENT SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: qwen-3-32b (Cerebras)\n")
        f.write(f"Task: Pick better translation (Naive vs Enhanced)\n")
        f.write(f"Chapters processed: {start_chapter}-{end_chapter}\n")
        f.write(f"Total chapters: {len(successful) + len(failed)}\n")
        f.write(f"Successful: {len(successful)}\n")
        f.write(f"Failed: {len(failed)}\n")
        f.write(f"Total time: {total_time:.1f}s\n")
        f.write(f"Average judgment time: {avg_judgment_time:.1f}s\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if successful:
            f.write("JUDGMENT RESULTS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Enhanced wins: {len(enhanced_wins)}/{len(successful)} ({len(enhanced_wins)/len(successful)*100:.1f}%)\n")
            f.write(f"Naive wins: {len(naive_wins)}/{len(successful)} ({len(naive_wins)/len(successful)*100:.1f}%)\n")
            f.write(f"Unclear: {len(unclear)}/{len(successful)} ({len(unclear)/len(successful)*100:.1f}%)\n\n")
            
            f.write("CONFIDENCE LEVELS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"High confidence: {len(high_conf)}/{len(successful)} ({len(high_conf)/len(successful)*100:.1f}%)\n")
            f.write(f"Medium confidence: {len(medium_conf)}/{len(successful)} ({len(medium_conf)/len(successful)*100:.1f}%)\n")
            f.write(f"Low confidence: {len(low_conf)}/{len(successful)} ({len(low_conf)/len(successful)*100:.1f}%)\n\n")
            
            f.write("CHAPTER-BY-CHAPTER RESULTS:\n")
            f.write("-" * 50 + "\n")
            f.write("Ch# | Winner    | Confidence | Reason\n")
            f.write("-" * 50 + "\n")
            for result in successful:
                ch = result["chapter"]
                winner = {"naive": "Naive", "enhanced": "Enhanced", "unknown": "Unclear"}[result["judgment"]["winner"]]
                conf = result["judgment"]["confidence"].title()
                reason = result["judgment"]["reason"][:40] + "..." if len(result["judgment"]["reason"]) > 40 else result["judgment"]["reason"]
                f.write(f"{ch:2d}  | {winner:9} | {conf:10} | {reason}\n")
        
        if failed:
            f.write(f"\nFAILED CHAPTERS:\n")
            f.write("-" * 20 + "\n")
            for result in failed:
                f.write(f"Chapter {result['chapter']}: {result.get('error', 'Unknown error')}\n")

def main():
    parser = argparse.ArgumentParser(description="Step 6c: LLM head-to-head judgment of naive vs enhanced translations")
    parser.add_argument("--start", type=int, default=1, help="Start chapter number")
    parser.add_argument("--end", type=int, default=3, help="End chapter number")
    parser.add_argument("--concurrent", type=int, default=3, help="Max concurrent requests")
    
    args = parser.parse_args()
    
    # Check API key
    if not os.getenv("CEREBRAS_API_KEY"):
        print("Error: CEREBRAS_API_KEY not found in environment")
        return
    
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
    
    print("Step 6c Configuration:")
    print(f"  Chapters: {args.start}-{args.end}")
    print(f"  Max concurrent: {args.concurrent}")
    print(f"  Model: qwen-3-32b (Cerebras)")
    print(f"  Task: Head-to-head comparison (Naive vs Enhanced)")
    print(f"  Output: ../results/evaluation/")
    print()
    
    # Run LLM judgment pipeline
    asyncio.run(run_llm_judgment_pipeline(args.start, args.end, args.concurrent))

if __name__ == "__main__":
    main()