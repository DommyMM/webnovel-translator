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
    enhanced_results_dir: str = "../results/enhanced"  # Your enhanced translations
    ground_truth_dir: str = "../data/chapters/ground_truth"
    chinese_chapters_dir: str = "../data/chapters/clean"  # For context
    terminology_db_file: str = "../data/terminology/extracted_terminology.json"
    raw_responses_dir: str = "../data/terminology/raw_responses"
    output_dir: str = "../data/terminology"
    start_chapter: int = 1
    end_chapter: int = 3
    model: str = "qwen-3-32b"  # Cerebras model
    temperature: float = 0.1   # Low temp for consistent extraction
    max_concurrent: int = 3    # Conservative for Cerebras

class SmartTerminologyExtractor:
    """Extract terminology differences using AI comparison"""
    
    def __init__(self, config: TerminologyConfig, chapter_num: int):
        self.config = config
        self.chapter_num = chapter_num
        self.client = Cerebras(api_key=os.getenv("CEREBRAS_API_KEY"))
        self.setup_directories()
        
    def setup_directories(self):
        Path(self.config.output_dir).mkdir(exist_ok=True, parents=True)
        Path(self.config.raw_responses_dir).mkdir(exist_ok=True, parents=True)
    
    def load_translations(self) -> tuple[str, str, str]:
        """Load Chinese original, enhanced translation, and ground truth"""
        
        # Load Chinese original (for context)
        chinese_file = Path(self.config.chinese_chapters_dir) / f"chapter_{self.chapter_num:04d}_cn.txt"
        if not chinese_file.exists():
            raise FileNotFoundError(f"Chinese chapter not found: {chinese_file}")
        
        with open(chinese_file, 'r', encoding='utf-8') as f:
            chinese_text = f.read().strip()
        
        # Load enhanced translation (your current best)
        enhanced_file = Path(self.config.enhanced_results_dir, "translations", f"chapter_{self.chapter_num:04d}_enhanced.txt")
        if not enhanced_file.exists():
            raise FileNotFoundError(f"Enhanced translation not found: {enhanced_file}")
        
        with open(enhanced_file, 'r', encoding='utf-8') as f:
            enhanced_text = f.read().strip()
            # Remove header if present
            if enhanced_text.startswith("Enhanced Translation"):
                lines = enhanced_text.split('\n')
                enhanced_text = '\n'.join(lines[3:]).strip()
        
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
        
        return chinese_text, enhanced_text, ground_truth
    
    def create_terminology_extraction_prompt(self, chinese_text: str, enhanced_text: str, ground_truth: str) -> str:
        """Create smart prompt for terminology extraction"""
        
        prompt = f"""<think>
I need to compare two English translations of a Chinese cultivation novel chapter and identify where they use different terminology for the same Chinese concepts. I should focus on terminology differences, not style or grammar differences.

Key areas to look for:
1. Character names - different romanizations or translations
2. Cultivation terms - realms, techniques, concepts
3. Titles and epithets - "Alchemy Emperor" vs "Pill God"
4. Techniques and arts - martial arts, cultivation methods
5. Places and organizations - sect names, location names
6. Items and medicines - pill names, artifact names

I should prefer the professional translation's terminology choices and identify where my translation differs.
</think>

You are an expert in Chinese cultivation novels. Compare these two English translations and extract terminology differences where they use different words for the same Chinese concepts.

CHINESE ORIGINAL (for context):
{chinese_text[:1000]}...

MY TRANSLATION:
{enhanced_text[:1000]}...

PROFESSIONAL REFERENCE (prefer this terminology):
{ground_truth[:1000]}...

Your task: Find terminology differences where MY TRANSLATION and PROFESSIONAL REFERENCE use different English words for the same Chinese concepts. Focus on names, cultivation terms, titles, techniques, and key concepts.

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

**Categories:** character, cultivation, title, technique, place, item

**Examples of what I'm looking for:**
```
丹帝 → Pill God (instead of Alchemy Emperor)
龙尘 → Long Chen (instead of Dragon Dust)  
金丹期 → Golden Core stage (instead of Golden Core realm)
```

Extract the most important terminology differences:"""
        
        return prompt
    
    async def extract_terminology_async(self, chinese_text: str, enhanced_text: str, ground_truth: str) -> List[Dict]:
        """Extract terminology using Cerebras AI"""
        
        prompt = self.create_terminology_extraction_prompt(chinese_text, enhanced_text, ground_truth)
        
        try:
            print(f"Chapter {self.chapter_num}: Analyzing terminology differences...")
            
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "You are an expert in Chinese cultivation novels specializing in terminology consistency. Extract terminology differences between translations with high precision."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=4096
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
        """Parse AI response into structured terminology entries"""
        
        # Remove thinking tags
        clean_response = re.sub(r'<think>.*?</think>', '', ai_response, flags=re.DOTALL).strip()
        
        terminology_entries = []
        
        # Look for the terminology differences section
        if "TERMINOLOGY_DIFFERENCES:" in clean_response:
            terminology_section = clean_response.split("TERMINOLOGY_DIFFERENCES:")[1]
        else:
            # Fallback: look for lines with the pattern
            terminology_section = clean_response
        
        # Parse lines with the pattern: "Chinese → Professional (instead of My)"
        lines = terminology_section.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('```'):
                continue
            
            # Pattern: "丹帝 → Pill God (instead of Alchemy Emperor)"
            match = re.match(r'^([^→]+)\s*→\s*([^(]+)\s*\(instead of ([^)]+)\)', line)
            if match:
                chinese_term = match.group(1).strip()
                professional_term = match.group(2).strip()
                my_term = match.group(3).strip()
                
                # Categorize the term
                category = self.categorize_term(chinese_term, professional_term)
                
                entry = {
                    "id": f"term_ch{self.chapter_num}_{len(terminology_entries)+1}",
                    "chinese_term": chinese_term,
                    "professional_term": professional_term,
                    "my_translation_term": my_term,
                    "category": category,
                    "frequency": 1,  # Will be updated when merging across chapters
                    "confidence": 0.8,  # High confidence from AI extraction
                    "context_example": line,
                    "chapters_seen": [self.chapter_num],
                    "created_at": datetime.now().isoformat()
                }
                
                terminology_entries.append(entry)
                print(f"  Extracted: {chinese_term} → {professional_term} (instead of {my_term})")
        
        return terminology_entries
    
    def categorize_term(self, chinese_term: str, professional_term: str) -> str:
        """Categorize the terminology type"""
        
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
        """Extract terminology for this specific chapter"""
        async with semaphore:
            print(f"Starting Chapter {self.chapter_num} terminology extraction...")
            
            try:
                # Load all three texts
                chinese_text, enhanced_text, ground_truth = self.load_translations()
                print(f"Chapter {self.chapter_num}: Loaded translations")
                
                # Extract terminology differences
                terminology_entries = await self.extract_terminology_async(chinese_text, enhanced_text, ground_truth)
                
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
    """Run smart terminology extraction across multiple chapters"""
    
    def __init__(self, config: TerminologyConfig):
        self.config = config
        self.setup_directories()
    
    def setup_directories(self):
        Path(self.config.output_dir).mkdir(exist_ok=True, parents=True)
        Path(self.config.raw_responses_dir).mkdir(exist_ok=True, parents=True)
    
    async def run_async_terminology_extraction(self):
        """Main pipeline to extract terminology differences asynchronously"""
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
        """Merge terminology from all chapters and save database"""
        
        # Merge identical terms across chapters
        merged_terms = {}
        
        for entry in all_terminology:
            chinese_term = entry["chinese_term"]
            professional_term = entry["professional_term"]
            
            # Create a key for merging (chinese + professional term)
            key = f"{chinese_term}→{professional_term}"
            
            if key not in merged_terms:
                merged_terms[key] = entry.copy()
            else:
                # Merge frequency and chapters
                merged_terms[key]["frequency"] += 1
                merged_terms[key]["chapters_seen"].extend(entry["chapters_seen"])
                merged_terms[key]["chapters_seen"] = sorted(list(set(merged_terms[key]["chapters_seen"])))
        
        # Convert to final database format
        terminology_db = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "total_terms": len(merged_terms),
                "chapters_processed": f"{self.config.start_chapter}-{self.config.end_chapter}",
                "extraction_method": "ai_comparison",
                "model_used": self.config.model
            },
            "terminology": {
                entry["chinese_term"]: {
                    "professional_term": entry["professional_term"],
                    "category": entry["category"],
                    "frequency": entry["frequency"],
                    "confidence": entry["confidence"],
                    "chapters_seen": entry["chapters_seen"],
                    "created_at": entry["created_at"]
                }
                for entry in merged_terms.values()
            }
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
            f.write(f"Chapters: {self.config.start_chapter}-{self.config.end_chapter}\n")
            f.write(f"Method: AI comparison with professional translations\n")
            f.write(f"Model: {self.config.model}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Group by category
            from collections import defaultdict
            by_category = defaultdict(list)
            for entry in merged_terms.values():
                by_category[entry["category"]].append(entry)
            
            for category, entries in sorted(by_category.items()):
                f.write(f"\n{category.upper()} ({len(entries)} terms):\n")
                f.write("-" * 40 + "\n")
                for entry in sorted(entries, key=lambda x: x["frequency"], reverse=True):
                    f.write(f"{entry['chinese_term']:15} → {entry['professional_term']:25} (freq: {entry['frequency']}, chapters: {entry['chapters_seen']})\n")
        
        print(f"Terminology database saved to: {self.config.terminology_db_file}")
        print(f"Readable format saved to: {readable_file}")
        
        # Show key results
        print(f"\nKEY TERMINOLOGY DIFFERENCES FOUND:")
        print("=" * 60)
        for i, entry in enumerate(list(merged_terms.values())[:10]):
            print(f"{entry['chinese_term']} → {entry['professional_term']} (instead of {entry['my_translation_term']})")

def main():
    parser = argparse.ArgumentParser(description="Smart Terminology Extraction Pipeline")
    parser.add_argument("--start", type=int, default=1, help="Start chapter")
    parser.add_argument("--end", type=int, default=3, help="End chapter")
    parser.add_argument("--chapter", type=int, help="Extract terminology from single chapter")
    parser.add_argument("--concurrent", type=int, default=2, help="Max concurrent requests")
    
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
        config.enhanced_results_dir,
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
        print("  1. python 4_enhanced_translate.py (for enhanced translations)")
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
        print("Smart Terminology Extraction Configuration:")
        print(f"  Model: {config.model}")
        print(f"  Temperature: {config.temperature}")
        print(f"  Max concurrent: {config.max_concurrent}")
        print(f"  Chapters: {config.start_chapter}-{config.end_chapter}")
        print(f"  Enhanced results: {config.enhanced_results_dir}")
        print(f"  Ground truth: {config.ground_truth_dir}")
        print(f"  Output: {config.terminology_db_file}")
        
        pipeline = SmartTerminologyPipeline(config)
        asyncio.run(pipeline.run_async_terminology_extraction())

if __name__ == "__main__":
    main()