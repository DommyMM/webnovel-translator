import os
import json
import re
import argparse
import asyncio
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv

load_dotenv()

@dataclass
class TerminologyEntry:
    chinese_term: str
    professional_term: str
    my_translation_term: str
    category: str  # "character", "cultivation", "title", "technique", "place"
    frequency: int
    confidence: float
    context_example: str
    chapters_seen: List[int]
    created_at: str

@dataclass
class TerminologyConfig:
    baseline_results_dir: str = "../results/baseline"  # Use baseline translations
    ground_truth_dir: str = "../data/chapters/ground_truth"
    chinese_chapters_dir: str = "../data/chapters/clean"  # For context
    terminology_db_file: str = "../data/terminology/extracted_terminology.json"
    raw_responses_dir: str = "../data/terminology/raw_responses"
    output_dir: str = "../data/terminology"
    start_chapter: int = 1
    end_chapter: int = 3
    model: str = "qwen-3-32b"  # Cerebras model
    temperature: float = 0.2   # Low temp for consistent extraction
    max_concurrent: int = 3    # Conservative for Cerebras

class SmartTerminologyExtractor:
    def __init__(self, config: TerminologyConfig, chapter_num: int):
        self.config = config
        self.chapter_num = chapter_num
        self.client = Cerebras(api_key=os.getenv("CEREBRAS_API_KEY"))
        self.setup_directories()
        
    def setup_directories(self):
        Path(self.config.output_dir).mkdir(exist_ok=True, parents=True)
        Path(self.config.raw_responses_dir).mkdir(exist_ok=True, parents=True)
    
    def load_translations(self) -> tuple[str, str, str]:
        # Load Chinese original (for context)
        chinese_file = Path(self.config.chinese_chapters_dir) / f"chapter_{self.chapter_num:04d}_cn.txt"
        if not chinese_file.exists():
            raise FileNotFoundError(f"Chinese chapter not found: {chinese_file}")
        
        with open(chinese_file, 'r', encoding='utf-8') as f:
            chinese_text = f.read().strip()
        
        # Load BASELINE translation (was enhanced)
        baseline_file = Path(self.config.baseline_results_dir, "translations", f"chapter_{self.chapter_num:04d}_deepseek.txt")
        if not baseline_file.exists():
            raise FileNotFoundError(f"Baseline translation not found: {baseline_file}")
        
        with open(baseline_file, 'r', encoding='utf-8') as f:
            baseline_text = f.read().strip()
            # Remove header if present
            if baseline_text.startswith("DeepSeek Translation"):
                lines = baseline_text.split('\n')
                baseline_text = '\n'.join(lines[2:]).strip()
        
        # Load professional ground truth
        truth_file = Path(self.config.ground_truth_dir) / f"chapter_{self.chapter_num:04d}_en.txt"
        if not truth_file.exists():
            raise FileNotFoundError(f"Ground truth not found: {truth_file}")
        
        with open(truth_file, 'r', encoding='utf-8') as f:
            ground_truth = f.read().strip()
            # Remove header if present
            lines = ground_truth.split('\n')
            if lines[0].startswith("Chapter"):
                ground_truth = '\n'.join(lines[3:]).strip()
        
        return chinese_text, baseline_text, ground_truth
    
    def create_terminology_extraction_prompt(self, chinese_text: str, baseline_text: str, ground_truth: str) -> str:
        prompt = f"""You are an expert in Chinese cultivation novels. Compare these two English translations and extract terminology differences where they use different words for the same Chinese concepts.
    
IMPORTANT: Do not include any thinking process, reasoning, or analysis in your response. Give only the final output.

CHINESE ORIGINAL (for context):
{chinese_text}

BASELINE TRANSLATION:
{baseline_text}

PROFESSIONAL REFERENCE (prefer this terminology):
{ground_truth}

Your task: Find terminology differences where BASELINE TRANSLATION and PROFESSIONAL REFERENCE use different English words for the same Chinese concepts. Focus on names, cultivation terms, titles, techniques, and key concepts.

**What to extract:**
- Character names (e.g., different romanizations)
- Cultivation realms and stages  
- Titles and epithets (e.g. "Alchemy Emperor" vs "Pill God")
- Technique/art names
- Sect/organization names
- Important items (pills, artifacts, etc.)

**What to ignore:**
- Style differences (sentence structure, flow)
- Minor word choice (said vs replied)
- Grammar differences

**Output format (extract 5-15 key terminology differences):**
```
TERMINOLOGY_DIFFERENCES:
[Chinese] → [Professional Term] (instead of [My Term])
[Chinese] → [Professional Term] (instead of [My Term])
...
```

**Examples:**
```
丹帝 → Pill God (instead of Alchemy Emperor)
龙尘 → Long Chen (instead of Dragon Dust)
龙夫人 → Madam Long (instead of Dragon Lady)
金丹期 → Golden Core stage (instead of Golden Core realm)
```

Extract the most important terminology differences:"""
        
        return prompt
    
    async def extract_terminology_async(self, chinese_text: str, baseline_text: str, ground_truth: str) -> List[Dict]:
        prompt = self.create_terminology_extraction_prompt(chinese_text, baseline_text, ground_truth)
        
        try:
            print(f"Chapter {self.chapter_num}: Analyzing terminology differences...")
            
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "You are a terminology extraction expert. Provide only the requested formatted output without showing your reasoning or thinking process."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=16382
            )
            
            ai_response = response.choices[0].message.content
            
            # Save raw response for debugging
            raw_file = Path(self.config.raw_responses_dir) / f"chapter_{self.chapter_num:04d}_terminology_raw.txt"
            with open(raw_file, 'w', encoding='utf-8') as f:
                f.write(f"Chapter {self.chapter_num} Terminology Extraction\n")
                f.write("=" * 60 + "\n\n")
                f.write("PROMPT:\n")
                f.write("-" * 30 + "\n")
                f.write(prompt[:500] + "...\n\n")
                f.write("RESPONSE:\n")  
                f.write("-" * 30 + "\n")
                f.write(ai_response)
            
            print(f"Chapter {self.chapter_num}: Raw response saved, parsing terminology...")
            
            # Parse the AI response
            terminology_entries = self.parse_terminology_response(ai_response)
            
            print(f"Chapter {self.chapter_num}: Extracted {len(terminology_entries)} terminology differences")
            
            return terminology_entries
            
        except Exception as e:
            print(f"Chapter {self.chapter_num}: Error extracting terminology: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def parse_terminology_response(self, ai_response: str) -> List[Dict]:
        print(f"Chapter {self.chapter_num}: Raw response length: {len(ai_response)} chars")
        
        terminology_entries = []
        
        # Look for the terminology differences section
        if "TERMINOLOGY_DIFFERENCES:" in ai_response:
            terminology_section = ai_response.split("TERMINOLOGY_DIFFERENCES:")[1]
            print(f"Chapter {self.chapter_num}: Found TERMINOLOGY_DIFFERENCES section")
        else:
            # Fallback: treat whole response as terminology section
            terminology_section = ai_response
            print(f"Chapter {self.chapter_num}: No section header found, using full response")
        
        # Parse lines with the pattern: "[Chinese] → [Professional] (instead of [My])"
        lines = terminology_section.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('```') or line.startswith('-'):
                continue
            
            # Try bracketed format first: [Chinese] → [Professional] (instead of [My])
            match = re.match(r'^\[([^\]]+)\]\s*→\s*\[([^\]]+)\]\s*\(instead of \[([^\]]+)\]\)', line)
            
            if not match:
                # Try mixed format: [Chinese] → Professional (instead of My)
                match = re.match(r'^\[([^\]]+)\]\s*→\s*([^(]+)\s*\(instead of ([^)]+)\)', line)
            
            if not match:
                # Try no brackets: Chinese → Professional (instead of My)
                match = re.match(r'^([^→]+)\s*→\s*([^(]+)\s*\(instead of ([^)]+)\)', line)
            
            if match:
                chinese_term = match.group(1).strip()
                professional_term = match.group(2).strip()
                my_term = match.group(3).strip()
                
                # Skip if any term is empty
                if not chinese_term or not professional_term or not my_term:
                    continue
                
                # Categorize the term
                category = self.categorize_term(chinese_term, professional_term)
                
                entry = {
                    "id": f"term_ch{self.chapter_num}_{len(terminology_entries)+1}",
                    "chinese_term": chinese_term,
                    "professional_term": professional_term,
                    "my_translation_term": my_term,
                    "category": category,
                    "frequency": 1,
                    "confidence": 0.8,
                    "context_example": line,
                    "chapters_seen": [self.chapter_num],
                    "created_at": datetime.now().isoformat()
                }
                
                terminology_entries.append(entry)
                print(f"  Extracted: {chinese_term} → {professional_term} (instead of {my_term})")
            else:
                # Log failed parsing for debugging
                if '→' in line and len(line) > 10:
                    print(f"  Failed to parse: {line}")
        
        print(f"Chapter {self.chapter_num}: Successfully parsed {len(terminology_entries)} entries")
        return terminology_entries
    
    def categorize_term(self, chinese_term: str, professional_term: str) -> str:
        # Character names (usually 2-3 chars, proper names)
        if (len(chinese_term) <= 3 and 
            re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?$', professional_term)):
            return "character"
        
        # Cultivation terms (realms, stages, etc.)
        if (chinese_term.endswith('期') or 
            any(word in professional_term.lower() for word in ['stage', 'realm', 'level', 'core', 'foundation'])):
            return "cultivation"
        
        # Titles and epithets
        if (chinese_term.endswith(('帝', '神', '王', '尊')) or
            any(word in professional_term.lower() for word in ['god', 'emperor', 'king', 'lord', 'master'])):
            return "title"
        
        # Techniques and arts
        if (chinese_term.endswith('诀') or
            any(word in professional_term.lower() for word in ['art', 'technique', 'method', 'skill'])):
            return "technique"
        
        # Places and organizations
        if (chinese_term.endswith(('宗', '门', '派')) or
            any(word in professional_term.lower() for word in ['sect', 'clan', 'family', 'palace'])):
            return "place"
        
        # Items (pills, artifacts, etc.)
        if ('丹' in chinese_term or
            any(word in professional_term.lower() for word in ['pill', 'pellet', 'medicine', 'artifact'])):
            return "item"
        
        return "general"
    
    async def extract_terminology_for_chapter(self, semaphore: asyncio.Semaphore) -> Optional[List[Dict]]:
        async with semaphore:
            print(f"Starting Chapter {self.chapter_num} terminology extraction...")
            
            try:
                # Load all three texts
                chinese_text, baseline_text, ground_truth = self.load_translations()
                print(f"Chapter {self.chapter_num}: Loaded translations")
                
                # Extract terminology differences
                terminology_entries = await self.extract_terminology_async(chinese_text, baseline_text, ground_truth)
                
                # Save individual chapter results
                chapter_file = Path(self.config.output_dir) / f"chapter_{self.chapter_num:04d}_terminology.json"
                chapter_data = {
                    "chapter": self.chapter_num,
                    "terminology_count": len(terminology_entries),
                    "terminology_entries": terminology_entries,
                    "timestamp": datetime.now().isoformat()
                }
                
                with open(chapter_file, 'w', encoding='utf-8') as f:
                    json.dump(chapter_data, f, indent=2, ensure_ascii=False)
                
                print(f"Chapter {self.chapter_num} complete: {len(terminology_entries)} terminology differences")
                
                return terminology_entries
                
            except Exception as e:
                print(f"Chapter {self.chapter_num}: Error processing: {e}")
                return None

class SmartTerminologyPipeline:
    def __init__(self, config: TerminologyConfig, rebuild: bool = False):
        self.config = config
        self.rebuild = rebuild
        self.setup_directories()
    
    def setup_directories(self):
        Path(self.config.output_dir).mkdir(exist_ok=True, parents=True)
        Path(self.config.raw_responses_dir).mkdir(exist_ok=True, parents=True)
    
    def load_existing_terminology_database(self) -> Dict:
        if self.rebuild:
            print("Rebuild flag: Starting fresh, ignoring existing terminology")
            return {"terminology": {}, "metadata": {}}
        
        if Path(self.config.terminology_db_file).exists():
            with open(self.config.terminology_db_file, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            print(f"Loaded {len(existing.get('terminology', {}))} existing terms for incremental learning")
            return existing
        else:
            print("No existing terminology found - starting fresh")
            return {"terminology": {}, "metadata": {}}
    
    def resolve_terminology_conflict(self, existing: Dict, new: Dict) -> Dict:
        # Prefer higher confidence
        if new["confidence"] > existing["confidence"]:
            print(f"  Conflict resolved: Using new term (higher confidence)")
            return new
        elif existing["confidence"] > new["confidence"]:
            print(f"  Conflict resolved: Keeping existing term (higher confidence)")
            return existing
        
        # If confidence is equal, prefer higher frequency
        if new["frequency"] > existing["frequency"]:
            print(f"  Conflict resolved: Using new term (higher frequency)")
            return new
        else:
            print(f"  Conflict resolved: Keeping existing term (higher frequency)")
            return existing
    
    def get_chapters_from_terminology(self, terminology: Dict) -> List[int]:
        chapters = set()
        for term_data in terminology.values():
            chapters_seen = term_data.get("chapters_seen", [])
            if isinstance(chapters_seen, list):
                chapters.update(chapters_seen)
        return sorted(list(chapters))
    
    async def run_async_terminology_extraction(self):
        print("Starting Smart Terminology Extraction Pipeline")
        print(f"Model: {self.config.model}")
        print(f"Processing chapters {self.config.start_chapter}-{self.config.end_chapter}")
        print(f"Max concurrent requests: {self.config.max_concurrent}")
        
        start_time = time.time()
        chapters = list(range(self.config.start_chapter, self.config.end_chapter + 1))
        
        # Create semaphore to limit concurrent requests (conservative for Cerebras)
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        # Create extractors for each chapter
        extractors = [SmartTerminologyExtractor(self.config, chapter_num) for chapter_num in chapters]
        
        # Create tasks for all chapters
        tasks = [extractor.extract_terminology_for_chapter(semaphore) for extractor in extractors]
        
        print(f"Launching {len(tasks)} concurrent terminology extraction tasks...")
        
        # Run all tasks concurrently
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        all_terminology = []
        failed_count = 0
        
        for i, result in enumerate(results_list):
            chapter_num = chapters[i]
            if isinstance(result, Exception):
                print(f"Chapter {chapter_num} failed with exception: {result}")
                failed_count += 1
            elif result is not None:
                all_terminology.extend(result)
            else:
                failed_count += 1
        
        extraction_time = time.time() - start_time
        print(f"Smart terminology extraction complete in {extraction_time:.1f}s")
        print(f"Successfully processed {len(chapters) - failed_count}/{len(chapters)} chapters")
        print(f"Total terminology differences extracted: {len(all_terminology)}")
        
        # Merge and save results
        print("Merging terminology database...")
        self.merge_and_save_terminology(all_terminology)
        
        total_time = time.time() - start_time
        print(f"Smart Terminology Extraction Pipeline Complete")
        print(f"Total time: {total_time:.1f}s")
        
        return all_terminology
    
    def merge_and_save_terminology(self, all_terminology: List[Dict]):
        # Load existing terminology database
        existing_db = self.load_existing_terminology_database()
        existing_terminology = existing_db.get("terminology", {})
        
        # Start with existing terminology
        merged_terms = {}
        conflicts_resolved = 0
        new_terms_added = 0
        
        # First, add all existing terms
        for chinese_term, term_data in existing_terminology.items():
            merged_terms[chinese_term] = term_data
        
        # Process new terminology entries
        for entry in all_terminology:
            chinese_term = entry["chinese_term"]
            professional_term = entry["professional_term"]
            
            # Create key for merging (chinese term)
            if chinese_term in merged_terms:
                # Conflict: same Chinese term, potentially different professional term
                existing_professional = merged_terms[chinese_term]["professional_term"]
                new_professional = professional_term
                
                if existing_professional != new_professional:
                    print(f"  Conflict: {chinese_term} → existing: '{existing_professional}' vs new: '{new_professional}'")
                    merged_terms[chinese_term] = self.resolve_terminology_conflict(
                        merged_terms[chinese_term], 
                        {
                            "professional_term": professional_term,
                            "category": entry["category"],
                            "frequency": entry["frequency"],
                            "confidence": entry["confidence"],
                            "chapters_seen": entry["chapters_seen"],
                            "created_at": entry["created_at"]
                        }
                    )
                    conflicts_resolved += 1
                else:
                    # Same translation, just merge frequency and chapters
                    merged_terms[chinese_term]["frequency"] += entry["frequency"]
                    existing_chapters = set(merged_terms[chinese_term]["chapters_seen"])
                    new_chapters = set(entry["chapters_seen"])
                    merged_terms[chinese_term]["chapters_seen"] = sorted(list(existing_chapters | new_chapters))
            else:
                # New term, add it
                merged_terms[chinese_term] = {
                    "professional_term": professional_term,
                    "category": entry["category"],
                    "frequency": entry["frequency"],
                    "confidence": entry["confidence"],
                    "chapters_seen": entry["chapters_seen"],
                    "created_at": entry["created_at"]
                }
                new_terms_added += 1
        
        # Calculate all contributing chapters
        all_contributing_chapters = self.get_chapters_from_terminology(merged_terms)
        current_batch_chapters = list(range(self.config.start_chapter, self.config.end_chapter + 1))
        
        # Convert to final database format
        previous_update_count = existing_db.get("metadata", {}).get("incremental_update_count", 0)
        terminology_db = {
            "metadata": {
                "created_at": existing_db.get("metadata", {}).get("created_at", datetime.now().isoformat()),
                "last_updated": datetime.now().isoformat(),
                "total_terms": len(merged_terms),
                "extraction_method": "ai_comparison_incremental",
                "model_used": self.config.model,
                "incremental_update_count": previous_update_count + 1,
                "current_batch_chapters": current_batch_chapters,
                "all_contributing_chapters": all_contributing_chapters,
                "new_terms_this_batch": new_terms_added,
                "conflicts_resolved_this_batch": conflicts_resolved
            },
            "terminology": merged_terms
        }
        
        # Save main database
        with open(self.config.terminology_db_file, 'w', encoding='utf-8') as f:
            json.dump(terminology_db, f, indent=2, ensure_ascii=False)
        
        # Save human-readable format
        readable_file = Path(self.config.output_dir) / "terminology_readable.txt"
        with open(readable_file, 'w', encoding='utf-8') as f:
            f.write("SMART EXTRACTED TERMINOLOGY DATABASE\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total terms: {len(merged_terms)}\n")
            f.write(f"Chapters: {all_contributing_chapters}\n")
            f.write(f"Method: AI comparison with professional translations\n")
            f.write(f"Model: {self.config.model}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Group by category
            from collections import defaultdict
            by_category = defaultdict(list)
            for chinese_term, term_data in merged_terms.items():
                by_category[term_data["category"]].append((chinese_term, term_data))
            
            for category, entries in sorted(by_category.items()):
                f.write(f"\n{category.upper()} ({len(entries)} terms):\n")
                f.write("-" * 40 + "\n")
                for chinese_term, term_data in sorted(entries, key=lambda x: x[1]["frequency"], reverse=True):
                    professional_term = term_data["professional_term"]
                    frequency = term_data["frequency"]
                    chapters_seen = term_data["chapters_seen"]
                    chapters_str = f"{min(chapters_seen)}-{max(chapters_seen)}" if len(chapters_seen) > 1 else str(chapters_seen[0])
                    f.write(f"{chinese_term:15} → {professional_term:25} (freq: {frequency}, chapters: {chapters_str})\n")
        
        print(f"Incremental merge complete:")
        print(f"  Previous terms: {len(existing_terminology)}")
        print(f"  New terms added: {new_terms_added}")
        print(f"  Conflicts resolved: {conflicts_resolved}")
        print(f"  Total terms now: {len(merged_terms)}")
        print(f"  Chapters covered: {all_contributing_chapters}")
        print(f"Terminology database saved to: {self.config.terminology_db_file}")
        print(f"Readable format saved to: {readable_file}")

def main():
    parser = argparse.ArgumentParser(description="Smart Terminology Extraction Pipeline with Incremental Learning")
    parser.add_argument("--start", type=int, default=1, help="Start chapter")
    parser.add_argument("--end", type=int, default=3, help="End chapter")
    parser.add_argument("--chapter", type=int, help="Extract terminology from single chapter")
    parser.add_argument("--concurrent", type=int, default=2, help="Max concurrent requests")
    parser.add_argument("--rebuild", action="store_true", help="Start fresh (ignore existing terminology)")
    
    args = parser.parse_args()
    
    # Check Cerebras API key
    if not os.getenv("CEREBRAS_API_KEY"):
        print("Error: CEREBRAS_API_KEY not found in environment")
        print("Please set your Cerebras API key in .env file")
        return
    
    config = TerminologyConfig(
        start_chapter=args.start,
        end_chapter=args.end,
        max_concurrent=args.concurrent
    )
    
    # Check if required directories exist
    required_dirs = [
        config.baseline_results_dir,
        config.ground_truth_dir,
        config.chinese_chapters_dir
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print("Error: Required directories not found:")
        for dir_path in missing_dirs:
            print(f"  - {dir_path}")
        print("\nPlease run the prerequisite steps:")
        print("  1. python 1_baseline_translate.py (for baseline translations)")
        return
    
    if args.chapter:
        # Single chapter mode
        print(f"Extracting terminology from single chapter: {args.chapter}")
        async def single_chapter():
            semaphore = asyncio.Semaphore(1)
            extractor = SmartTerminologyExtractor(config, args.chapter)
            result = await extractor.extract_terminology_for_chapter(semaphore)
            if result:
                print(f"Chapter {args.chapter} processed: {len(result)} terminology differences")
            else:
                print(f"Chapter {args.chapter} processing failed")
        
        asyncio.run(single_chapter())
    else:
        # Process all chapters
        mode = "REBUILD" if args.rebuild else "INCREMENTAL"
        print(f"Smart Terminology Extraction Configuration ({mode} MODE):")
        print(f"  Model: {config.model}")
        print(f"  Temperature: {config.temperature}")
        print(f"  Max concurrent: {config.max_concurrent}")
        print(f"  Chapters: {config.start_chapter}-{config.end_chapter}")
        print(f"  Baseline results: {config.baseline_results_dir}")
        print(f"  Ground truth: {config.ground_truth_dir}")
        print(f"  Output: {config.terminology_db_file}")
        print(f"  Rebuild mode: {args.rebuild}")
        
        pipeline = SmartTerminologyPipeline(config, rebuild=args.rebuild)
        asyncio.run(pipeline.run_async_terminology_extraction())

if __name__ == "__main__":
    main()