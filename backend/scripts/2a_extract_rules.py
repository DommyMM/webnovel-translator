import os
import time
import json
import re
import argparse
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass, asdict
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

@dataclass
class TranslationRule:
    id: str
    rule_type: str  # "terminology", "style", "grammar", "cultural"
    description: str
    chinese_pattern: str
    english_pattern: str
    examples: List[Dict]
    confidence: float
    usage_count: int
    success_rate: float
    created_at: str
    last_used: str

@dataclass
class ComparisonMetrics:
    chapter_num: int
    similarity_score: float
    word_overlap: float
    length_ratio: float
    key_differences: List[str]
    extracted_rules: List[str]
    timestamp: str

@dataclass
class LearningConfig:
    deepseek_results_dir: str = "../results/baseline"
    ground_truth_dir: str = "../data/chapters/ground_truth"
    rules_database_file: str = "../data/rules/extracted_raw.json"
    output_dir: str = "../results/analysis"
    start_chapter: int = 1
    end_chapter: int = 3
    model: str = "deepseek-chat"
    temperature: float = 1.0
    max_tokens: int = 8192
    base_url: str = "https://api.deepseek.com"
    min_confidence: float = 0.7
    max_rules_per_comparison: int = 5
    max_concurrent: int = 10

class AsyncRuleExtractor:
    """Extract rules from a single chapter asynchronously"""
    
    def __init__(self, config: LearningConfig, chapter_num: int):
        self.config = config
        self.chapter_num = chapter_num
        self.client = AsyncOpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=self.config.base_url
        )
        self.setup_directories()
        
    def setup_directories(self):
        Path(self.config.output_dir).mkdir(exist_ok=True)
        for subdir in ["comparisons", "rules", "analytics", "raw_responses", "temp_rules"]:
            Path(self.config.output_dir, subdir).mkdir(exist_ok=True)
    
    def load_translations(self) -> tuple[str, str, str]:      # Load DeepSeek translation, ground truth, and original Chinese text
        # DeepSeek translation
        deepseek_file = Path(self.config.deepseek_results_dir, "translations", f"chapter_{self.chapter_num:04d}_deepseek.txt")
        if not deepseek_file.exists():
            raise FileNotFoundError(f"DeepSeek translation not found: {deepseek_file}")
        
        with open(deepseek_file, 'r', encoding='utf-8') as f:
            deepseek_text = f.read().strip()
            # Remove the header line if present
            if deepseek_text.startswith("DeepSeek Translation"):
                lines = deepseek_text.split('\n')
                deepseek_text = '\n'.join(lines[2:]).strip()
        
        # Ground truth
        truth_file = Path(self.config.ground_truth_dir, f"chapter_{self.chapter_num:04d}_en.txt")
        if not truth_file.exists():
            raise FileNotFoundError(f"Ground truth not found: {truth_file}")
        
        with open(truth_file, 'r', encoding='utf-8') as f:
            ground_truth = f.read().strip()
            # Remove chapter header if present
            lines = ground_truth.split('\n')
            if lines[0].startswith("Chapter"):
                ground_truth = '\n'.join(lines[3:]).strip()
        
        # Original Chinese (for context)
        chinese_file = Path("../data/chapters/clean", f"chapter_{self.chapter_num:04d}_cn.txt")
        chinese_text = ""
        if chinese_file.exists():
            with open(chinese_file, 'r', encoding='utf-8') as f:
                chinese_text = f.read().strip()
        
        return deepseek_text, ground_truth, chinese_text
    
    async def analyze_differences(self, deepseek_text: str, ground_truth: str, chinese_text: str) -> tuple[List[Dict], float]:        
        prompt = f"""You are a translation expert analyzing two English translations of a Chinese cultivation novel. Extract STYLE and STRUCTURAL rules (NOT terminology) that would make MY TRANSLATION more like the PROFESSIONAL REFERENCE.

NOTE: Ignore specific word/term choices - focus only on style, structure, and flow patterns.

ORIGINAL CHINESE:
{chinese_text}

MY TRANSLATION (needs improvement):
{deepseek_text}

PROFESSIONAL REFERENCE (target quality):
{ground_truth}

Analyze where MY TRANSLATION differs from the PROFESSIONAL REFERENCE and extract 3-5 core translation rules focused on STYLE and STRUCTURE only. Focus on:

1. **Style preferences** - How the professional reference handles tone, voice, formality, register
2. **Structural patterns** - How the professional reference organizes sentences, paragraphs, dialogue
3. **Flow and rhythm** - How the professional reference creates natural English flow
4. **Cultural adaptation** - How the professional reference handles cultural elements and context

IGNORE: Specific terminology choices, character names, technique names, realm names, etc.

For each rule, provide:
RULE_TYPE: [style|structure|flow|cultural]
PATTERN: What stylistic/structural approach should be used to match professional quality  
EXAMPLE: Specific difference showing how structure/style differs (not terminology)
CONFIDENCE: [high|medium|low]

Focus only on abstract stylistic and structural principles that will apply broadly across many chapters. Do NOT extract rules about specific word choices or terminology - only about HOW to write and structure the translation."""
        
        try:
            print(f"Chapter {self.chapter_num}: Making AI analysis call...")
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "You are an expert translator specializing in Chinese cultivation novels. Analyze translation differences to extract actionable improvement rules."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            analysis = response.choices[0].message.content
            
            # Save raw response for debugging
            raw_response_file = Path(self.config.output_dir, "raw_responses", f"chapter_{self.chapter_num:04d}_raw_analysis.txt")
            with open(raw_response_file, 'w', encoding='utf-8') as f:
                f.write(f"Chapter {self.chapter_num} Raw AI Analysis\n")
                f.write("=" * 50 + "\n\n")
                f.write(analysis)
            
            print(f"Chapter {self.chapter_num}: Response saved, parsing rules...")
            
            # Try to parse rules
            rules = self.parse_rule_analysis(analysis)
            print(f"Chapter {self.chapter_num}: Successfully parsed {len(rules)} rules")
            
            # Calculate overall similarity
            similarity = self.calculate_similarity(deepseek_text, ground_truth)
            
            return rules, similarity
            
        except Exception as e:
            print(f"Chapter {self.chapter_num}: Error in rule analysis: {e}")
            import traceback
            traceback.print_exc()
            return [], 0.0
    
    def parse_rule_analysis(self, analysis_text: str) -> List[Dict]:        # Parse AI analysis text into structured rules
        print(f"Chapter {self.chapter_num}: Attempting to parse rules...")
        rules = []
        
        # Split analysis into sections - try multiple approaches
        sections = []
        
        # Try splitting by numbered lists first
        numbered_sections = re.split(r'\n(?=\d+\.)', analysis_text)
        if len(numbered_sections) > 1:
            sections = numbered_sections
            print(f"Chapter {self.chapter_num}: Found {len(sections)} numbered sections")
        else:
            # Try splitting by headers with asterisks
            header_sections = re.split(r'\n(?=\*\*[^*]+\*\*)', analysis_text)
            if len(header_sections) > 1:
                sections = header_sections
                print(f"Chapter {self.chapter_num}: Found {len(sections)} header sections")
            else:
                # Try splitting by "RULE_TYPE" markers
                rule_sections = re.split(r'\n(?=RULE_TYPE)', analysis_text, flags=re.IGNORECASE)
                if len(rule_sections) > 1:
                    sections = rule_sections
                    print(f"Chapter {self.chapter_num}: Found {len(sections)} RULE_TYPE sections")
                else:
                    # Fallback: treat entire text as one section
                    sections = [analysis_text]
                    print(f"Chapter {self.chapter_num}: Using entire text as one section")
        
        for i, section in enumerate(sections):
            section = section.strip()
            if not section or len(section) < 30:
                continue
                
            print(f"Chapter {self.chapter_num}: Processing section {i+1}")
            print(f"Section preview: {section[:150]}...")
            
            current_rule = {
                "id": f"rule_ch{self.chapter_num}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}",
                "created_at": datetime.now().isoformat(),
                "usage_count": 0,
                "success_rate": 0.0,
                "last_used": None,
                "rule_type": "general",  # default
                "description": "",
                "confidence": 0.7,  # default
                "examples": []
            }
            
            # Extract rule type - try multiple patterns
            type_patterns = [
                r'RULE_TYPE:\s*\[([^\]]+)\]',
                r'RULE_TYPE:\s*([^\n]+)',
                r'Type:\s*([^\n]+)',
                r'\*\*([^*]*(?:terminology|style|structure|cultural)[^*]*)\*\*'
            ]
            
            # Update valid rule types (remove terminology)
            valid_rule_types = ['style', 'structure', 'flow', 'cultural', 'general']
            
            for pattern in type_patterns:
                match = re.search(pattern, section, re.IGNORECASE)
                if match:
                    rule_type = match.group(1).strip().lower()
                    # Extract just the key type if it's in a longer phrase
                    for key_type in valid_rule_types:
                        if key_type in rule_type:
                            current_rule["rule_type"] = key_type
                            break
                    else:
                        current_rule["rule_type"] = "general"  # fallback
                    print(f"  Found rule_type: {current_rule['rule_type']}")
                    break
            
            # Extract pattern/description
            desc_patterns = [
                r'PATTERN:\s*(.+?)(?=EXAMPLE:|CONFIDENCE:|$)',
                r'Pattern:\s*(.+?)(?=Example:|Confidence:|$)',
                r'\d+\.\s*\*\*[^*]+\*\*\s*(.+?)(?=Example:|Confidence:|RULE_TYPE|$)',
                r'\*\*[^*]+\*\*\s*(.+?)(?=Example:|Confidence:|RULE_TYPE|$)'
            ]
            
            for pattern in desc_patterns:
                match = re.search(pattern, section, re.DOTALL | re.IGNORECASE)
                if match:
                    description = match.group(1).strip()
                    # Clean up the description
                    description = re.sub(r'\n+', ' ', description)
                    description = re.sub(r'\s+', ' ', description)
                    current_rule["description"] = description
                    print(f"  Found description: {description[:100]}...")
                    break
            
            # If no formal pattern found, extract first substantial sentence
            if not current_rule["description"]:
                sentences = re.split(r'[.!?]+', section)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if len(sentence) > 30 and not any(x in sentence.lower() for x in ['rule_type', 'pattern:', 'example:', 'confidence:']):
                        current_rule["description"] = sentence
                        print(f"  Using sentence as description: {sentence[:100]}...")
                        break
            
            # Extract confidence
            conf_patterns = [
                r'CONFIDENCE:\s*\[([^\]]+)\]',
                r'CONFIDENCE:\s*([^\n]+)',
                r'Confidence:\s*([^\n]+)'
            ]
            
            for pattern in conf_patterns:
                match = re.search(pattern, section, re.IGNORECASE)
                if match:
                    conf_text = match.group(1).strip().lower()
                    confidence_map = {"high": 0.9, "medium": 0.7, "low": 0.5}
                    current_rule["confidence"] = confidence_map.get(conf_text, 0.7)
                    print(f"  Found confidence: {current_rule['confidence']}")
                    break
            
            # Extract example
            example_patterns = [
                r'EXAMPLE:\s*(.+?)(?=CONFIDENCE:|RULE_TYPE|$)',
                r'Example:\s*(.+?)(?=Confidence:|RULE_TYPE|$)'
            ]
            
            for pattern in example_patterns:
                match = re.search(pattern, section, re.DOTALL | re.IGNORECASE)
                if match:
                    example_text = match.group(1).strip()
                    current_rule["examples"] = [{"text": example_text, "source": "comparison"}]
                    print(f"  Found example: {example_text[:100]}...")
                    break
            
            # Validate rule
            if self.is_valid_rule(current_rule):
                rules.append(current_rule)
                print(f"  Valid rule added")
            else:
                print(f"  Invalid rule (missing: {[f for f in ['rule_type', 'description', 'confidence'] if not current_rule.get(f)]})")
        
        print(f"Chapter {self.chapter_num}: Final parsing result: {len(rules)} valid rules")
        return rules[:self.config.max_rules_per_comparison]
    
    def is_valid_rule(self, rule: Dict) -> bool:
        required_fields = ["rule_type", "description", "confidence"]
        if not all(field in rule and rule[field] for field in required_fields):
            return False
        
        # Reject rules that seem to be about terminology
        description = rule.get("description", "").lower()
        terminology_keywords = [
            "use the term", "translate as", "should be called", "refer to as",
            "name should be", "terminology", "word choice", "term choice"
        ]
        
        for keyword in terminology_keywords:
            if keyword in description:
                print(f"  Rejected terminology rule: {description[:100]}...")
                return False
        
        return True
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    async def extract_rules_for_chapter(self, semaphore: asyncio.Semaphore) -> Dict:        # Extract rules for this specific chapter and return results
        async with semaphore:  # Limit concurrent requests
            print(f"Starting Chapter {self.chapter_num} analysis...")
            
            try:
                # Load translations
                deepseek_text, ground_truth, chinese_text = self.load_translations()
                print(f"Chapter {self.chapter_num}: Loaded translations - DeepSeek: {len(deepseek_text)} chars, Truth: {len(ground_truth)} chars")
                
                # Analyze differences and extract rules
                extracted_rules, similarity = await self.analyze_differences(deepseek_text, ground_truth, chinese_text)
                
                # Filter high-confidence rules
                high_conf_rules = [r for r in extracted_rules if r.get("confidence", 0) >= self.config.min_confidence]
                
                # Calculate metrics
                word_overlap = self.calculate_similarity(deepseek_text, ground_truth)
                length_ratio = len(deepseek_text) / len(ground_truth) if ground_truth else 0
                
                metrics = {
                    "chapter_num": self.chapter_num,
                    "similarity_score": similarity,
                    "word_overlap": word_overlap,
                    "length_ratio": length_ratio,
                    "key_differences": [],
                    "extracted_rules": [r.get("id", "") for r in extracted_rules],
                    "timestamp": datetime.now().isoformat()
                }
                
                # Save individual chapter results
                chapter_result = {
                    "chapter": self.chapter_num,
                    "metrics": metrics,
                    "extracted_rules": extracted_rules,
                    "high_conf_rules": high_conf_rules,
                    "texts": {
                        "deepseek_length": len(deepseek_text),
                        "ground_truth_length": len(ground_truth),
                        "deepseek_preview": deepseek_text[:200],
                        "ground_truth_preview": ground_truth[:200]
                    },
                    "timestamp": datetime.now().isoformat()
                }
                
                # Save to individual file for merging later
                temp_file = Path(self.config.output_dir, "temp_rules", f"chapter_{self.chapter_num:04d}_rules.json")
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(chapter_result, f, indent=2, ensure_ascii=False)
                
                print(f"Chapter {self.chapter_num} complete: Extracted {len(extracted_rules)} rules ({len(high_conf_rules)} high-confidence)")
                
                return chapter_result
                
            except Exception as e:
                print(f"Chapter {self.chapter_num}: Error processing: {e}")
                import traceback
                traceback.print_exc()
                return None

class AsyncRuleLearningPipeline:
    def __init__(self, config: LearningConfig, rebuild: bool = False):
        self.config = config
        self.rebuild = rebuild
        self.setup_directories()
        Path(self.config.rules_database_file).parent.mkdir(parents=True, exist_ok=True)
    
    def setup_directories(self):
        Path(self.config.output_dir).mkdir(exist_ok=True)
        for subdir in ["comparisons", "rules", "analytics", "raw_responses", "temp_rules"]:
            Path(self.config.output_dir, subdir).mkdir(exist_ok=True)
    
    async def run_async_extraction(self):      # Main pipeline to run the rule learning process asynchronously
        print("Starting Async Rule Extraction Pipeline")
        print(f"Processing chapters {self.config.start_chapter}-{self.config.end_chapter} concurrently")
        print(f"Max concurrent requests: {self.config.max_concurrent}")
        
        start_time = time.time()
        chapters = list(range(self.config.start_chapter, self.config.end_chapter + 1))
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        # Create extractors for each chapter
        extractors = [AsyncRuleExtractor(self.config, chapter_num) for chapter_num in chapters]
        
        # Create tasks for all chapters
        tasks = [extractor.extract_rules_for_chapter(semaphore) for extractor in extractors]
        
        print(f"Launching {len(tasks)} concurrent rule extraction tasks...")
        
        # Run all tasks concurrently
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        results = {}
        failed_count = 0
        
        for i, result in enumerate(results_list):
            chapter_num = chapters[i]
            if isinstance(result, Exception):
                print(f"Chapter {chapter_num} failed with exception: {result}")
                failed_count += 1
            elif result is not None:
                results[chapter_num] = result
            else:
                failed_count += 1
        
        extraction_time = time.time() - start_time
        print(f"Async extraction complete in {extraction_time:.1f}s")
        print(f"Successfully processed {len(results)}/{len(chapters)} chapters")
        print(f"Failed chapters: {failed_count}")
        
        # Merge all results
        print("Merging results...")
        self.merge_all_results(results)
        
        total_time = time.time() - start_time
        print(f"Async Rule Learning Pipeline Complete")
        print(f"Total time: {total_time:.1f}s")
        
        return results
    
    def load_existing_rules_database(self) -> Dict:
        if self.rebuild:
            print("Rebuild flag: Starting fresh, ignoring existing rules")
            return {"rules": [], "metadata": {}}
        
        if Path(self.config.rules_database_file).exists():
            with open(self.config.rules_database_file, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            print(f"Loaded {len(existing.get('rules', []))} existing rules for incremental learning")
            return existing
        else:
            print("No existing rules found - starting fresh")
            return {"rules": [], "metadata": {}}
    
    def get_chapters_from_rules(self, rules: List[Dict]) -> List[int]:
        chapters = set()
        for rule in rules:
            # Extract chapter number from rule ID (format: rule_ch{num}_...)
            rule_id = rule.get("id", "")
            if "rule_ch" in rule_id:
                try:
                    chapter_part = rule_id.split("rule_ch")[1].split("_")[0]
                    chapters.add(int(chapter_part))
                except (IndexError, ValueError):
                    pass
        return sorted(list(chapters))

    def merge_all_results(self, results: Dict):        # Merge results from all chapters into final database
        
        # Load existing rules if they exist
        existing_db = self.load_existing_rules_database()
        
        # Start with existing rules instead of empty list
        merged_database = {
            "rules": existing_db.get("rules", []).copy(),  # Start with existing
            "metadata": existing_db.get("metadata", {})
        }
        
        all_metrics = []
        
        # Add new rules from current extraction
        new_rules_count = 0
        for chapter_num in sorted(results.keys()):
            result = results[chapter_num]
            
            # Add high-confidence rules to merged database
            high_conf_rules = result.get("high_conf_rules", [])
            merged_database["rules"].extend(high_conf_rules)
            new_rules_count += len(high_conf_rules)
            
            # Collect metrics
            all_metrics.append(result["metrics"])
            
            # Save individual comparison file
            comparison_file = Path(self.config.output_dir, "comparisons", f"chapter_{chapter_num:04d}_analysis.json")
            with open(comparison_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Calculate all chapters that have contributed to this rule database
        all_contributing_chapters = self.get_chapters_from_rules(merged_database["rules"])
        current_batch_chapters = list(range(self.config.start_chapter, self.config.end_chapter + 1))
        
        # Update metadata with incremental learning info
        previous_update_count = existing_db.get("metadata", {}).get("incremental_update_count", 0)
        merged_database["metadata"].update({
            "total_rules": len(merged_database["rules"]),
            "created_at": existing_db.get("metadata", {}).get("created_at", datetime.now().isoformat()),
            "last_updated": datetime.now().isoformat(),
            "extraction_method": "async_incremental",
            "incremental_update_count": previous_update_count + 1,
            "current_batch_chapters": current_batch_chapters,
            "all_contributing_chapters": all_contributing_chapters,
            "new_rules_this_batch": new_rules_count,
            "max_concurrent": self.config.max_concurrent
        })
        
        # Save merged rules database
        with open(self.config.rules_database_file, 'w', encoding='utf-8') as f:
            json.dump(merged_database, f, indent=2, ensure_ascii=False)
        
        # Save analytics
        self.save_analytics(merged_database, all_metrics)
        
        # Clean up temp files
        temp_dir = Path(self.config.output_dir, "temp_rules")
        if temp_dir.exists():
            for temp_file in temp_dir.glob("*.json"):
                temp_file.unlink()
        
        print(f"Incremental merge complete:")
        print(f"  Previous rules: {len(existing_db.get('rules', []))}")
        print(f"  New rules added: {new_rules_count}")
        print(f"  Total rules now: {len(merged_database['rules'])}")
        print(f"  Chapters covered: {all_contributing_chapters}")
        print(f"Rules database saved to: {self.config.rules_database_file}")
    
    def save_analytics(self, merged_database: Dict, all_metrics: List):        # Save final analytics
        analytics_file = Path(self.config.output_dir, "analytics", "learning_analytics.json")
        
        high_conf_rules = [r for r in merged_database["rules"] if r.get("confidence", 0) >= 0.8]
        rule_types = {}
        for rule in merged_database["rules"]:
            rule_type = rule.get("rule_type", "unknown")
            rule_types[rule_type] = rule_types.get(rule_type, 0) + 1
        
        analytics = {
            "config": asdict(self.config),
            "summary": {
                "chapters_analyzed": len(all_metrics),
                "total_rules_extracted": len(merged_database["rules"]),
                "high_confidence_rules": len(high_conf_rules),
                "rule_types_distribution": rule_types,
                "avg_similarity": sum(m["similarity_score"] for m in all_metrics) / len(all_metrics) if all_metrics else 0,
                "extraction_method": "async",
                "max_concurrent": self.config.max_concurrent
            },
            "comparison_metrics": all_metrics,
            "rules_database": merged_database,
            "generated_at": datetime.now().isoformat()
        }
        
        with open(analytics_file, 'w', encoding='utf-8') as f:
            json.dump(analytics, f, indent=2, ensure_ascii=False)
        
        print(f"Analytics saved to: {analytics_file}")

def main():
    parser = argparse.ArgumentParser(description="Async Rule Extraction Pipeline with Incremental Learning")
    parser.add_argument("--chapter", type=int, help="Process single chapter (for individual extraction)")
    parser.add_argument("--start", type=int, default=1, help="Start chapter for async processing")
    parser.add_argument("--end", type=int, default=3, help="End chapter for async processing")
    parser.add_argument("--concurrent", type=int, default=3, help="Max concurrent requests")
    parser.add_argument("--rebuild", action="store_true", help="Start fresh (ignore existing rules)")
    
    args = parser.parse_args()
    
    # Check API key
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("Error: DEEPSEEK_API_KEY not found in environment")
        return
    
    config = LearningConfig(
        start_chapter=args.start,
        end_chapter=args.end,
        model="deepseek-reasoner",
        temperature=1.0,
        min_confidence=0.7,
        max_concurrent=args.concurrent
    )
    
    # Check if required directories exist
    required_dirs = [config.deepseek_results_dir, config.ground_truth_dir]
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            print(f"Error: Required directory not found: {dir_path}")
            return
    
    if args.chapter:
        # Single chapter mode (for individual processing)
        print(f"Processing single chapter: {args.chapter}")
        async def single_chapter():
            semaphore = asyncio.Semaphore(1)
            extractor = AsyncRuleExtractor(config, args.chapter)
            result = await extractor.extract_rules_for_chapter(semaphore)
            if result:
                print(f"Chapter {args.chapter} processed successfully")
            else:
                print(f"Chapter {args.chapter} processing failed")
        
        asyncio.run(single_chapter())
    else:
        # Async mode (process all chapters)
        mode = "REBUILD" if args.rebuild else "INCREMENTAL"
        print(f"Async Rule Extraction Configuration ({mode} MODE):")
        print(f"  DeepSeek results: {config.deepseek_results_dir}")
        print(f"  Ground truth: {config.ground_truth_dir}")
        print(f"  Chapters: {config.start_chapter}-{config.end_chapter}")
        print(f"  Min confidence: {config.min_confidence}")
        print(f"  Model: {config.model}")
        print(f"  Max concurrent: {config.max_concurrent}")
        print(f"  Rebuild mode: {args.rebuild}")
        
        pipeline = AsyncRuleLearningPipeline(config, rebuild=args.rebuild)
        asyncio.run(pipeline.run_async_extraction())

if __name__ == "__main__":
    main()