import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class ChapterMetrics:
    """Track metrics for each chapter processing"""
    chapter_num: int
    raw_similarity: float
    enhanced_similarity: Optional[float]
    improvement_delta: float
    new_rules_extracted: int
    total_rules_applied: int
    processing_time: float
    translation_time: float
    rule_extraction_time: float
    timestamp: str

@dataclass
class PipelineConfig:
    """Configuration for the Phase 1 pipeline"""
    # Paths
    chinese_chapters_dir: str = "clean_chapters"
    english_chapters_dir: str = "translated_chapters" 
    output_dir: str = "phase1_results"
    rules_dir: str = "rules_database"
    
    # Processing
    start_chapter: int = 1
    end_chapter: int = 10
    
    # Model settings
    model: str = "llama-3.3-70b"
    temperature: float = 0.1
    max_tokens: int = 4000
    
    # Rule management
    rule_confidence_threshold: float = 0.7
    max_rules_in_context: int = 15

class Phase1Pipeline:
    """Main orchestrator for Phase 1 translation pipeline"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.client = Cerebras(api_key=os.getenv("CEREBRAS_API_KEY"))
        self.setup_directories()
        self.metrics: List[ChapterMetrics] = []
        self.rules_database: Dict = {"rules": [], "metadata": {"last_updated": None, "total_rules": 0}}
        
    def setup_directories(self):
        """Create necessary output directories"""
        for dir_path in [self.config.output_dir, self.config.rules_dir]:
            Path(dir_path).mkdir(exist_ok=True)
            
        # Create subdirectories for organization
        for subdir in ["raw_translations", "enhanced_translations", "comparisons", "analytics"]:
            Path(self.config.output_dir, subdir).mkdir(exist_ok=True)
    
    def load_chapter_files(self, chapter_num: int) -> tuple[str, str]:
        """Load Chinese and English files for a chapter"""
        cn_file = Path(self.config.chinese_chapters_dir) / f"chapter_{chapter_num:04d}_cn.txt"
        en_file = Path(self.config.english_chapters_dir) / f"chapter_{chapter_num:04d}_en.txt"
        
        if not cn_file.exists():
            raise FileNotFoundError(f"Chinese chapter not found: {cn_file}")
        if not en_file.exists():
            raise FileNotFoundError(f"English chapter not found: {en_file}")
            
        with open(cn_file, 'r', encoding='utf-8') as f:
            chinese_text = f.read().strip()
        with open(en_file, 'r', encoding='utf-8') as f:
            english_truth = f.read().strip()
            
        return chinese_text, english_truth
    
    def create_translation_prompt(self, chinese_text: str, rules: List[Dict] = None) -> str:
        """Create translation prompt with optional rules"""
        base_prompt = """You are an expert translator specializing in Chinese web novels, particularly cultivation/xianxia genre. 
Translate the following Chinese chapter to English with these priorities:

1. Maintain narrative flow and readability
2. Keep character names consistent
3. Preserve cultural context and technical terminology
4. Use natural, engaging English prose"""

        if rules and len(rules) > 0:
            rules_text = "\n\nIMPORTANT TRANSLATION RULES to follow:\n"
            for i, rule in enumerate(rules[:self.config.max_rules_in_context], 1):
                rules_text += f"{i}. {rule.get('description', 'N/A')}\n"
                if rule.get('examples'):
                    rules_text += f"   Example: {rule['examples'][0]}\n"
            base_prompt += rules_text

        base_prompt += f"\n\nChinese text to translate:\n\n{chinese_text}"
        return base_prompt
    
    def translate_chapter(self, chinese_text: str, rules: List[Dict] = None) -> tuple[str, float]:
        """Translate a chapter using Cerebras with optional rules"""
        start_time = time.time()
        
        prompt = self.create_translation_prompt(chinese_text, rules)
        
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            translation = response.choices[0].message.content
            translation_time = time.time() - start_time
            
            return translation, translation_time
            
        except Exception as e:
            print(f"❌ Translation failed: {e}")
            return "", time.time() - start_time
    
    def calculate_similarity(self, translation: str, ground_truth: str) -> float:
        """Calculate similarity between translation and ground truth"""
        # Placeholder for now - will implement proper similarity in similarity.py
        # For Phase 1A, using simple length-based similarity as proof of concept
        
        trans_words = set(translation.lower().split())
        truth_words = set(ground_truth.lower().split())
        
        if not truth_words:
            return 0.0
            
        intersection = len(trans_words & truth_words)
        union = len(trans_words | truth_words)
        
        # Jaccard similarity as placeholder
        similarity = intersection / union if union > 0 else 0.0
        return min(similarity * 2, 1.0)  # Scale up for reasonable scores
    
    def extract_rules_from_comparison(self, translation: str, ground_truth: str, chinese_text: str) -> tuple[List[Dict], float]:
        """Use AI to extract translation rules from comparison"""
        start_time = time.time()
        
        comparison_prompt = f"""Compare these two English translations of the same Chinese text and extract specific translation rules.

Chinese Original:
{chinese_text[:500]}...

My Translation:
{translation[:1000]}...

Professional Translation (Ground Truth):
{ground_truth[:1000]}...

Analyze the differences and extract 3-5 specific translation rules that would improve future translations. 
Focus on:
1. Character name consistency
2. Technical terminology (cultivation terms, titles, etc.)
3. Cultural context preservation
4. Style and tone preferences

Format each rule as:
RULE_TYPE: [Character_Names|Cultivation_Terms|Cultural_Context|Style|Structure]
PATTERN: Specific pattern to follow
CONFIDENCE: High/Medium/Low
EXAMPLE: Concrete example from the texts

Be specific and actionable."""

        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": comparison_prompt}],
                model=self.config.model,
                temperature=0.1,
                max_tokens=1000
            )
            
            rule_analysis = response.choices[0].message.content
            extraction_time = time.time() - start_time
            
            # Parse the response into structured rules (simplified for Phase 1A)
            rules = self.parse_rule_analysis(rule_analysis)
            
            return rules, extraction_time
            
        except Exception as e:
            print(f"❌ Rule extraction failed: {e}")
            return [], time.time() - start_time
    
    def parse_rule_analysis(self, analysis_text: str) -> List[Dict]:
        """Parse AI rule analysis into structured rules"""
        # Simplified parsing for Phase 1A - will enhance in later phases
        rules = []
        lines = analysis_text.split('\n')
        
        current_rule = {}
        for line in lines:
            line = line.strip()
            if line.startswith('RULE_TYPE:'):
                if current_rule:
                    rules.append(current_rule)
                current_rule = {
                    "type": line.replace('RULE_TYPE:', '').strip(),
                    "confidence": 0.5,  # Default
                    "created_at": datetime.now().isoformat(),
                    "usage_count": 0
                }
            elif line.startswith('PATTERN:'):
                current_rule["description"] = line.replace('PATTERN:', '').strip()
            elif line.startswith('CONFIDENCE:'):
                conf_text = line.replace('CONFIDENCE:', '').strip().lower()
                current_rule["confidence"] = {"high": 0.8, "medium": 0.6, "low": 0.4}.get(conf_text, 0.5)
            elif line.startswith('EXAMPLE:'):
                current_rule["examples"] = [line.replace('EXAMPLE:', '').strip()]
        
        if current_rule:
            rules.append(current_rule)
            
        return rules
    
    def get_high_confidence_rules(self) -> List[Dict]:
        """Get rules above confidence threshold for translation"""
        return [rule for rule in self.rules_database["rules"] 
                if rule.get("confidence", 0) >= self.config.rule_confidence_threshold]
    
    def update_rules_database(self, new_rules: List[Dict]):
        """Add new rules to the database"""
        for rule in new_rules:
            rule["id"] = len(self.rules_database["rules"]) + 1
            self.rules_database["rules"].append(rule)
        
        self.rules_database["metadata"]["last_updated"] = datetime.now().isoformat()
        self.rules_database["metadata"]["total_rules"] = len(self.rules_database["rules"])
        
        # Save to file
        rules_file = Path(self.config.rules_dir) / "rules_database.json"
        with open(rules_file, 'w', encoding='utf-8') as f:
            json.dump(self.rules_database, f, indent=2, ensure_ascii=False)
    
    def save_chapter_results(self, chapter_num: int, chinese_text: str, raw_translation: str, 
                           enhanced_translation: str, ground_truth: str, extracted_rules: List[Dict]):
        """Save all chapter processing results"""
        # Save translations
        raw_file = Path(self.config.output_dir, "raw_translations", f"chapter_{chapter_num:04d}_raw.txt")
        with open(raw_file, 'w', encoding='utf-8') as f:
            f.write(raw_translation)
            
        if enhanced_translation:
            enhanced_file = Path(self.config.output_dir, "enhanced_translations", f"chapter_{chapter_num:04d}_enhanced.txt")
            with open(enhanced_file, 'w', encoding='utf-8') as f:
                f.write(enhanced_translation)
        
        # Save comparison analysis
        comparison_file = Path(self.config.output_dir, "comparisons", f"chapter_{chapter_num:04d}_comparison.json")
        comparison_data = {
            "chapter": chapter_num,
            "chinese_length": len(chinese_text),
            "raw_translation_length": len(raw_translation),
            "enhanced_translation_length": len(enhanced_translation) if enhanced_translation else 0,
            "ground_truth_length": len(ground_truth),
            "extracted_rules": extracted_rules,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, indent=2, ensure_ascii=False)
    
    def process_chapter(self, chapter_num: int) -> ChapterMetrics:
        """Process a single chapter through the complete pipeline"""
        print(f"\n{'='*60}")
        print(f"📖 Processing Chapter {chapter_num}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Step 1: Load files
        try:
            chinese_text, ground_truth = self.load_chapter_files(chapter_num)
            print(f"✓ Loaded files - CN: {len(chinese_text)} chars, EN: {len(ground_truth)} chars")
        except FileNotFoundError as e:
            print(f"❌ {e}")
            return None
        
        # Step 2: Raw translation (no rules)
        print("🔄 Creating raw translation...")
        raw_translation, translation_time = self.translate_chapter(chinese_text)
        raw_similarity = self.calculate_similarity(raw_translation, ground_truth)
        print(f"✓ Raw translation complete - Similarity: {raw_similarity:.3f}")
        
        # Step 3: Extract rules from comparison
        print("🔍 Extracting rules from comparison...")
        extracted_rules, extraction_time = self.extract_rules_from_comparison(
            raw_translation, ground_truth, chinese_text
        )
        print(f"✓ Extracted {len(extracted_rules)} new rules")
        
        # Step 4: Update rules database
        self.update_rules_database(extracted_rules)
        
        # Step 5: Enhanced translation (with rules) - only if we have rules
        enhanced_translation = ""
        enhanced_similarity = None
        improvement_delta = 0.0
        
        high_confidence_rules = self.get_high_confidence_rules()
        if high_confidence_rules:
            print(f"🔄 Creating enhanced translation with {len(high_confidence_rules)} rules...")
            enhanced_translation, _ = self.translate_chapter(chinese_text, high_confidence_rules)
            enhanced_similarity = self.calculate_similarity(enhanced_translation, ground_truth)
            improvement_delta = enhanced_similarity - raw_similarity
            print(f"✓ Enhanced translation complete - Similarity: {enhanced_similarity:.3f} (Δ{improvement_delta:+.3f})")
        else:
            print("⚠️ No high-confidence rules yet, skipping enhanced translation")
        
        # Step 6: Save results
        self.save_chapter_results(chapter_num, chinese_text, raw_translation, 
                                enhanced_translation, ground_truth, extracted_rules)
        
        # Step 7: Create metrics
        total_time = time.time() - start_time
        metrics = ChapterMetrics(
            chapter_num=chapter_num,
            raw_similarity=raw_similarity,
            enhanced_similarity=enhanced_similarity,
            improvement_delta=improvement_delta,
            new_rules_extracted=len(extracted_rules),
            total_rules_applied=len(high_confidence_rules),
            processing_time=total_time,
            translation_time=translation_time,
            rule_extraction_time=extraction_time,
            timestamp=datetime.now().isoformat()
        )
        
        self.metrics.append(metrics)
        
        print(f"✅ Chapter {chapter_num} complete in {total_time:.1f}s")
        return metrics
    
    def save_final_analytics(self):
        """Save comprehensive analytics for the entire run"""
        analytics_file = Path(self.config.output_dir, "analytics", "phase1_analytics.json")
        
        analytics = {
            "config": asdict(self.config),
            "summary": {
                "chapters_processed": len(self.metrics),
                "total_rules_extracted": self.rules_database["metadata"]["total_rules"],
                "avg_raw_similarity": sum(m.raw_similarity for m in self.metrics) / len(self.metrics) if self.metrics else 0,
                "avg_enhanced_similarity": sum(m.enhanced_similarity for m in self.metrics if m.enhanced_similarity) / 
                                         len([m for m in self.metrics if m.enhanced_similarity]) if any(m.enhanced_similarity for m in self.metrics) else 0,
                "avg_improvement": sum(m.improvement_delta for m in self.metrics) / len(self.metrics) if self.metrics else 0
            },
            "chapter_metrics": [asdict(m) for m in self.metrics],
            "rules_database": self.rules_database,
            "generated_at": datetime.now().isoformat()
        }
        
        with open(analytics_file, 'w', encoding='utf-8') as f:
            json.dump(analytics, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Analytics saved to: {analytics_file}")
    
    def run_pipeline(self):
        """Execute the complete Phase 1 pipeline"""
        print("🚀 Starting Phase 1 Translation Pipeline")
        print(f"📚 Processing chapters {self.config.start_chapter}-{self.config.end_chapter}")
        print(f"🤖 Using model: {self.config.model}")
        
        start_time = time.time()
        
        for chapter_num in range(self.config.start_chapter, self.config.end_chapter + 1):
            try:
                metrics = self.process_chapter(chapter_num)
                if metrics:
                    # Show progress summary
                    print(f"\n📈 Progress Summary:")
                    print(f"   Raw Similarity: {metrics.raw_similarity:.3f}")
                    if metrics.enhanced_similarity:
                        print(f"   Enhanced Similarity: {metrics.enhanced_similarity:.3f}")
                        print(f"   Improvement: {metrics.improvement_delta:+.3f}")
                    print(f"   Total Rules: {self.rules_database['metadata']['total_rules']}")
                    
            except Exception as e:
                print(f"❌ Failed to process chapter {chapter_num}: {e}")
                continue
        
        # Final analytics
        total_time = time.time() - start_time
        print(f"\n🎉 Phase 1 Pipeline Complete!")
        print(f"⏱️  Total time: {total_time:.1f}s")
        print(f"📊 Chapters processed: {len(self.metrics)}")
        
        if self.metrics:
            avg_raw = sum(m.raw_similarity for m in self.metrics) / len(self.metrics)
            enhanced_metrics = [m for m in self.metrics if m.enhanced_similarity]
            avg_enhanced = sum(m.enhanced_similarity for m in enhanced_metrics) / len(enhanced_metrics) if enhanced_metrics else 0
            
            print(f"📈 Average raw similarity: {avg_raw:.3f}")
            if avg_enhanced > 0:
                print(f"📈 Average enhanced similarity: {avg_enhanced:.3f}")
                print(f"📈 Overall improvement: {avg_enhanced - avg_raw:+.3f}")
        
        self.save_final_analytics()

def main():
    """Main entry point"""
    # Default configuration - can be modified here
    config = PipelineConfig(
        start_chapter=1,
        end_chapter=3,  # Start small for testing
        model="llama-3.3-70b",
        temperature=0.1
    )
    
    # Check API key
    if not os.getenv("CEREBRAS_API_KEY"):
        print("❌ CEREBRAS_API_KEY environment variable not set")
        return
    
    # Initialize and run pipeline
    pipeline = Phase1Pipeline(config)
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()