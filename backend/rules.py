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
    deepseek_results_dir: str = "deepseek_results"
    ground_truth_dir: str = "translated_chapters"
    rules_database_file: str = "rules_database.json"
    output_dir: str = "learning_results"
    
    start_chapter: int = 1
    end_chapter: int = 3
    
    # AI settings for rule extraction
    model: str = "deepseek-chat"
    temperature: float = 1.0  # Match Data Cleaning recommendation
    max_tokens: int = 8192
    base_url: str = "https://api.deepseek.com"
    
    # Rule filtering
    min_confidence: float = 0.7
    max_rules_per_comparison: int = 5

class RuleLearningPipeline:    
    def __init__(self, config: LearningConfig):
        self.config = config
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=self.config.base_url
        )
        self.setup_directories()
        self.rules_database = self.load_rules_database()
        self.comparison_metrics: List[ComparisonMetrics] = []
        
    def setup_directories(self):
        Path(self.config.output_dir).mkdir(exist_ok=True)
        for subdir in ["comparisons", "rules", "analytics", "raw_responses"]:
            Path(self.config.output_dir, subdir).mkdir(exist_ok=True)
    
    def load_rules_database(self) -> Dict:  # Load or initialize the rules database
        rules_file = Path(self.config.rules_database_file)
        if rules_file.exists():
            with open(rules_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {
                "rules": [],
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "total_rules": 0,
                    "last_updated": None
                }
            }
    
    def save_rules_database(self):
        self.rules_database["metadata"]["last_updated"] = datetime.now().isoformat()
        self.rules_database["metadata"]["total_rules"] = len(self.rules_database["rules"])
        
        with open(self.config.rules_database_file, 'w', encoding='utf-8') as f:
            json.dump(self.rules_database, f, indent=2, ensure_ascii=False)
    
    def load_translations(self, chapter_num: int) -> tuple[str, str, str]:      # Load DeepSeek translation, ground truth, and original Chinese text
        # DeepSeek translation
        deepseek_file = Path(self.config.deepseek_results_dir, "translations", f"chapter_{chapter_num:04d}_deepseek.txt")
        if not deepseek_file.exists():
            raise FileNotFoundError(f"DeepSeek translation not found: {deepseek_file}")
        
        with open(deepseek_file, 'r', encoding='utf-8') as f:
            deepseek_text = f.read().strip()
            # Remove the header line if present
            if deepseek_text.startswith("DeepSeek Translation"):
                lines = deepseek_text.split('\n')
                deepseek_text = '\n'.join(lines[2:]).strip()  # Skip title and empty line
        
        # Ground truth
        truth_file = Path(self.config.ground_truth_dir, f"chapter_{chapter_num:04d}_en.txt")
        if not truth_file.exists():
            raise FileNotFoundError(f"Ground truth not found: {truth_file}")
        
        with open(truth_file, 'r', encoding='utf-8') as f:
            ground_truth = f.read().strip()
            # Remove chapter header if present
            lines = ground_truth.split('\n')
            if lines[0].startswith("Chapter"):
                ground_truth = '\n'.join(lines[3:]).strip()  # Skip title, word count, separator
        
        # Original Chinese (for context)
        chinese_file = Path("clean_chapters", f"chapter_{chapter_num:04d}_cn.txt")
        chinese_text = ""
        if chinese_file.exists():
            with open(chinese_file, 'r', encoding='utf-8') as f:
                chinese_text = f.read().strip()
        
        return deepseek_text, ground_truth, chinese_text
    
    def analyze_differences(self, deepseek_text: str, ground_truth: str, chinese_text: str, chapter_num: int) -> tuple[List[Dict], float]:        
        prompt = f"""You are a translation expert analyzing two English translations of a Chinese cultivation novel. Extract only the most significant, abstract translation rules that would improve future work.

ORIGINAL CHINESE:
{chinese_text}

MY TRANSLATION:
{deepseek_text}

PROFESSIONAL REFERENCE:
{ground_truth}

Extract 3-5 core translation rules that represent the most significant patterns for improvement. Focus on:

1. **High-impact terminology** - key terms that appear frequently
2. **Fundamental style differences** - tone, formality, voice
3. **Structural patterns** - how sentences/paragraphs are constructed
4. **Cultural adaptation principles** - how concepts are localized

For each rule, provide:
RULE_TYPE: [terminology|style|structure|cultural]
PATTERN: One clear, actionable principle
EXAMPLE: Specific before/after from the texts
CONFIDENCE: [high|medium|low]

Be concise. Focus only on abstract principles that will apply broadly across many chapters, not specific word choices."""
        
        try:
            print("Making AI analysis call...")
            response = self.client.chat.completions.create(
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
            raw_response_file = Path(self.config.output_dir, "raw_responses", f"chapter_{chapter_num:04d}_raw_analysis.txt")
            with open(raw_response_file, 'w', encoding='utf-8') as f:
                f.write(f"Chapter {chapter_num} Raw AI Analysis\n")
                f.write("=" * 50 + "\n\n")
                f.write(analysis)
            
            print(f"Raw response saved to: {raw_response_file}")
            print(f"Response length: {len(analysis)} characters")
            print(f"Response preview (first 300 chars):\n{analysis[:300]}...\n")
            
            # Try to parse rules
            rules = self.parse_rule_analysis(analysis)
            print(f"Successfully parsed {len(rules)} rules")
            
            # Calculate overall similarity
            similarity = self.calculate_similarity(deepseek_text, ground_truth)
            
            return rules, similarity
            
        except Exception as e:
            print(f"Error in rule analysis: {e}")
            import traceback
            traceback.print_exc()
            return [], 0.0
    
    def parse_rule_analysis(self, analysis_text: str) -> List[Dict]:        # Parse AI analysis text into structured rules
        print("Attempting to parse rules...")
        rules = []
        
        # Split analysis into sections - try multiple approaches
        sections = []
        
        # Try splitting by numbered lists first
        numbered_sections = re.split(r'\n(?=\d+\.)', analysis_text)
        if len(numbered_sections) > 1:
            sections = numbered_sections
            print(f"Found {len(sections)} numbered sections")
        else:
            # Try splitting by headers with asterisks
            header_sections = re.split(r'\n(?=\*\*[^*]+\*\*)', analysis_text)
            if len(header_sections) > 1:
                sections = header_sections
                print(f"Found {len(sections)} header sections")
            else:
                # Try splitting by "RULE_TYPE" markers
                rule_sections = re.split(r'\n(?=RULE_TYPE)', analysis_text, flags=re.IGNORECASE)
                if len(rule_sections) > 1:
                    sections = rule_sections
                    print(f"Found {len(sections)} RULE_TYPE sections")
                else:
                    # Fallback: treat entire text as one section
                    sections = [analysis_text]
                    print("Using entire text as one section")
        
        for i, section in enumerate(sections):
            section = section.strip()
            if not section or len(section) < 30:
                continue
                
            print(f"\nProcessing section {i+1}:")
            print(f"Section preview: {section[:150]}...")
            
            current_rule = {
                "id": f"rule_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}",
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
            
            for pattern in type_patterns:
                match = re.search(pattern, section, re.IGNORECASE)
                if match:
                    rule_type = match.group(1).strip().lower()
                    # Extract just the key type if it's in a longer phrase
                    for key_type in ['terminology', 'style', 'structure', 'cultural']:
                        if key_type in rule_type:
                            current_rule["rule_type"] = key_type
                            break
                    else:
                        current_rule["rule_type"] = rule_type
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
                print(f"  ✓ Valid rule added")
            else:
                print(f"  ✗ Invalid rule (missing: {[f for f in ['rule_type', 'description', 'confidence'] if not current_rule.get(f)]})")
        
        print(f"\nFinal parsing result: {len(rules)} valid rules")
        return rules[:self.config.max_rules_per_comparison]
    
    def is_valid_rule(self, rule: Dict) -> bool:
        required_fields = ["rule_type", "description", "confidence"]
        return all(field in rule and rule[field] for field in required_fields)
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def add_rules_to_database(self, new_rules: List[Dict]):
        for rule in new_rules:
            if rule.get("confidence", 0) >= self.config.min_confidence:
                self.rules_database["rules"].append(rule)
        
        self.save_rules_database()
    
    def process_chapter_comparison(self, chapter_num: int) -> ComparisonMetrics:
        print(f"\nAnalyzing Chapter {chapter_num}")
        print("-" * 40)
        
        try:
            # Load translations
            deepseek_text, ground_truth, chinese_text = self.load_translations(chapter_num)
            print(f"Loaded translations - DeepSeek: {len(deepseek_text)} chars, Truth: {len(ground_truth)} chars")
            
            # Analyze differences and extract rules
            print("Extracting rules from comparison...")
            extracted_rules, similarity = self.analyze_differences(deepseek_text, ground_truth, chinese_text, chapter_num)
            
            print(f"Extracted {len(extracted_rules)} rules, similarity: {similarity:.3f}")
            
            # Add high-confidence rules to database
            high_conf_rules = [r for r in extracted_rules if r.get("confidence", 0) >= self.config.min_confidence]
            self.add_rules_to_database(high_conf_rules)
            
            # Calculate metrics
            word_overlap = self.calculate_similarity(deepseek_text, ground_truth)
            length_ratio = len(deepseek_text) / len(ground_truth) if ground_truth else 0
            
            metrics = ComparisonMetrics(
                chapter_num=chapter_num,
                similarity_score=similarity,
                word_overlap=word_overlap,
                length_ratio=length_ratio,
                key_differences=[],
                extracted_rules=[r.get("id", "") for r in extracted_rules],
                timestamp=datetime.now().isoformat()
            )
            
            self.comparison_metrics.append(metrics)
            
            # Save comparison results
            self.save_comparison_results(chapter_num, deepseek_text, ground_truth, extracted_rules, metrics)
            
            print(f"Added {len(high_conf_rules)} high-confidence rules to database")
            print(f"Total rules in database: {len(self.rules_database['rules'])}")
            
            return metrics
            
        except Exception as e:
            print(f"Error processing chapter {chapter_num}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_comparison_results(self, chapter_num: int, deepseek_text: str, ground_truth: str, rules: List[Dict], metrics: ComparisonMetrics):
        comparison_file = Path(self.config.output_dir, "comparisons", f"chapter_{chapter_num:04d}_analysis.json")
        
        comparison_data = {
            "chapter": chapter_num,
            "metrics": asdict(metrics),
            "extracted_rules": rules,
            "texts": {
                "deepseek_length": len(deepseek_text),
                "ground_truth_length": len(ground_truth),
                "deepseek_preview": deepseek_text[:200],
                "ground_truth_preview": ground_truth[:200]
            },
            "timestamp": datetime.now().isoformat()
        }
        
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, indent=2, ensure_ascii=False)
    
    def run_learning_pipeline(self):      # Main pipeline to run the rule learning process
        print("Starting Rule Learning Pipeline")
        print(f"Analyzing chapters {self.config.start_chapter}-{self.config.end_chapter}")
        print(f"DeepSeek results: {self.config.deepseek_results_dir}")
        print(f"Ground truth: {self.config.ground_truth_dir}")
        
        start_time = time.time()
        
        for chapter_num in range(self.config.start_chapter, self.config.end_chapter + 1):
            try:
                metrics = self.process_chapter_comparison(chapter_num)
                if metrics:
                    print(f"Progress: {len(self.comparison_metrics)}/{self.config.end_chapter - self.config.start_chapter + 1} chapters")
            except Exception as e:
                print(f"Failed to process chapter {chapter_num}: {e}")
                continue
        
        # Final summary
        total_time = time.time() - start_time
        print(f"\nRule Learning Pipeline Complete")
        print(f"Total time: {total_time:.1f}s")
        print(f"Chapters analyzed: {len(self.comparison_metrics)}")
        print(f"Total rules learned: {len(self.rules_database['rules'])}")
        
        if self.comparison_metrics:
            avg_similarity = sum(m.similarity_score for m in self.comparison_metrics) / len(self.comparison_metrics)
            print(f"Average similarity: {avg_similarity:.3f}")
        
        self.save_final_analytics()
    
    def save_final_analytics(self):
        analytics_file = Path(self.config.output_dir, "analytics", "learning_analytics.json")
        
        high_conf_rules = [r for r in self.rules_database["rules"] if r.get("confidence", 0) >= 0.8]
        rule_types = {}
        for rule in self.rules_database["rules"]:
            rule_type = rule.get("rule_type", "unknown")
            rule_types[rule_type] = rule_types.get(rule_type, 0) + 1
        
        analytics = {
            "config": asdict(self.config),
            "summary": {
                "chapters_analyzed": len(self.comparison_metrics),
                "total_rules_extracted": len(self.rules_database["rules"]),
                "high_confidence_rules": len(high_conf_rules),
                "rule_types_distribution": rule_types,
                "avg_similarity": sum(m.similarity_score for m in self.comparison_metrics) / len(self.comparison_metrics) if self.comparison_metrics else 0
            },
            "comparison_metrics": [asdict(m) for m in self.comparison_metrics],
            "rules_database": self.rules_database,
            "generated_at": datetime.now().isoformat()
        }
        
        with open(analytics_file, 'w', encoding='utf-8') as f:
            json.dump(analytics, f, indent=2, ensure_ascii=False)
        
        print(f"Analytics saved to: {analytics_file}")

def main():
    config = LearningConfig(
        start_chapter=1,
        end_chapter=3,
        model="deepseek-chat",
        temperature=1.0,
        min_confidence=0.7
    )
    
    # Check API key
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("Error: DEEPSEEK_API_KEY not found in environment")
        return
    
    # Check if required directories exist
    required_dirs = [config.deepseek_results_dir, config.ground_truth_dir]
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            print(f"Error: Required directory not found: {dir_path}")
            return
    
    print("Learning Configuration:")
    print(f"  DeepSeek results: {config.deepseek_results_dir}")
    print(f"  Ground truth: {config.ground_truth_dir}")
    print(f"  Chapters: {config.start_chapter}-{config.end_chapter}")
    print(f"  Min confidence: {config.min_confidence}")
    print(f"  Model: {config.model}")
    
    # Initialize and run pipeline
    pipeline = RuleLearningPipeline(config)
    pipeline.run_learning_pipeline()

if __name__ == "__main__":
    main()