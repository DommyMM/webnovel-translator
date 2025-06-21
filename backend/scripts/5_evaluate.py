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
class EvaluationMetrics:
    chapter_num: int
    baseline_score: float
    enhanced_score: float
    improvement: float
    professional_baseline: float = 100.0
    
    # Detailed breakdowns
    flow_score: float = 0.0
    character_voice_score: float = 0.0
    clarity_score: float = 0.0
    genre_score: float = 0.0
    
    evaluator_comments: str = ""
    evaluation_time: float = 0.0
    timestamp: str = ""

@dataclass
class EvaluationConfig:
    baseline_results_dir: str = "../results/baseline"
    enhanced_results_dir: str = "../results/enhanced"
    ground_truth_dir: str = "../data/chapters/ground_truth"
    evaluation_output_dir: str = "../results/evaluation"
    start_chapter: int = 1
    end_chapter: int = 3
    # Use DeepSeek for evaluation but with different settings for unbiased scoring
    evaluator_model: str = "deepseek-chat"
    temperature: float = 0.2  # Low temp for consistent, objective scoring
    max_tokens: int = 2048
    # For DeepSeek API
    base_url: str = "https://api.deepseek.com"

class TranslationEvaluator:     # Objective evaluation system for translation quality improvement
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        
        # Use DeepSeek for evaluation with different settings for objectivity
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=self.config.base_url
        )
        
        self.setup_directories()
        self.evaluation_metrics: List[EvaluationMetrics] = []
    
    def setup_directories(self):
        Path(self.config.evaluation_output_dir).mkdir(exist_ok=True)
        for subdir in ["scores", "comparisons", "analytics", "reports"]:
            Path(self.config.evaluation_output_dir, subdir).mkdir(exist_ok=True)
    
    def load_translations(self, chapter_num: int) -> tuple[str, str, str]:       # Load baseline, enhanced, and ground truth translations
        
        # Load baseline translation (DeepSeek raw)
        baseline_file = Path(self.config.baseline_results_dir, "translations", f"chapter_{chapter_num:04d}_deepseek.txt")
        if not baseline_file.exists():
            raise FileNotFoundError(f"Baseline translation not found: {baseline_file}")
        
        with open(baseline_file, 'r', encoding='utf-8') as f:
            baseline_text = f.read().strip()
            # Remove header if present
            if baseline_text.startswith("DeepSeek Translation"):
                lines = baseline_text.split('\n')
                baseline_text = '\n'.join(lines[2:]).strip()
        
        # Load enhanced translation (with rules applied)
        enhanced_file = Path(self.config.enhanced_results_dir, "translations", f"chapter_{chapter_num:04d}_enhanced.txt")
        if not enhanced_file.exists():
            raise FileNotFoundError(f"Enhanced translation not found: {enhanced_file}")
        
        with open(enhanced_file, 'r', encoding='utf-8') as f:
            enhanced_text = f.read().strip()
            # Remove header if present
            if enhanced_text.startswith("Enhanced Translation"):
                lines = enhanced_text.split('\n')
                enhanced_text = '\n'.join(lines[3:]).strip()  # Skip title, rules count, empty line
        
        # Load professional ground truth (100% baseline)
        truth_file = Path(self.config.ground_truth_dir, f"chapter_{chapter_num:04d}_en.txt")
        if not truth_file.exists():
            raise FileNotFoundError(f"Ground truth not found: {truth_file}")
        
        with open(truth_file, 'r', encoding='utf-8') as f:
            ground_truth = f.read().strip()
            # Remove chapter header if present
            lines = ground_truth.split('\n')
            if lines[0].startswith("Chapter"):
                ground_truth = '\n'.join(lines[3:]).strip()
        
        return baseline_text, enhanced_text, ground_truth
    
    def create_evaluation_prompt(self, translation: str, ground_truth: str, translation_type: str) -> str:     # Create evaluation prompt for AI scoring
        
        prompt = f"""You are a Western reader who enjoys cultivation novels. Your job is to rate how much you'd enjoy reading this translation compared to the professional version.
    
PROFESSIONAL REFERENCE (your 100% gold standard):
{ground_truth}

{translation_type.upper()} TRANSLATION (rate this version):
{translation}

**Your Task**: Rate this translation as a Western reader who wants to enjoy the story.

**What You Care About** (as a Western cultivation novel reader):
- **Natural English Flow**: Does it read smoothly like a real English novel?
- **Character Personality**: Do characters feel real and consistent?  
- **Story Enjoyment**: Can you follow the action and get invested?
- **Proper Cultivation Terms**: Do terms like "Spiritual Strength" feel right?
- **Western Style**: Written for Western readers colloquially, not literal translation

**What You DON'T Care About**:
- Perfect word-for-word accuracy if meaning is clear
- Minor differences that don't affect story flow
- Academic translation precision over readability

**Key Question**: "Would I rather read this version or the professional version?"

**Scoring Scale**:
- 95-100%: Perfect - reads like a professional English novel, no issues unless you nitpick
- 85-95%: Excellent - reads like professional English, would prefer this over many fan translations
- 70-84%: Very good - smooth reading with minor artifacts, fully enjoyable
- 55-69%: Good - readable with some awkward phrasing, but story flows well
- 40-54%: Acceptable - clearly machine translated but understandable, some choppy sections
- 25-39%: Poor - heavy machine translation artifacts, difficult to follow story
- 10-24%: Very poor - broken English, major comprehension issues
- 0-9%: Unreadable - incomprehensible, completely failed translation

**Rate these aspects**:
1. **Overall Enjoyment**: How much would you enjoy this vs professional?
2. **Reading Flow**: Does it read smoothly?
3. **Character Voice**: Do characters feel real and consistent?
4. **Story Clarity**: Can you follow what's happening?
5. **Genre Feel**: Does it feel like a proper cultivation novel?

**Response Format**:
OVERALL: [score]%
FLOW: [score]%
CHARACTER: [score]%  
CLARITY: [score]%
GENRE: [score]%
COMMENTS: [2-3 sentences about what works well or needs improvement for reader enjoyment]"""
    
        return prompt
    
    def evaluate_translation(self, translation: str, ground_truth: str, translation_type: str) -> Dict:      # Get AI evaluation scores for a translation
        
        start_time = time.time()
        prompt = self.create_evaluation_prompt(translation, ground_truth, translation_type)
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.evaluator_model,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator of English translations for Western readers. Rate translations objectively based on readability and enjoyment for the target audience."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            evaluation_text = response.choices[0].message.content
            evaluation_time = time.time() - start_time
            
            # Parse the evaluation
            scores = self.parse_evaluation_response(evaluation_text)
            scores["evaluation_time"] = evaluation_time
            scores["raw_response"] = evaluation_text
            
            return scores
            
        except Exception as e:
            print(f"Evaluation failed for {translation_type}: {str(e)}")
            return {
                "overall_score": 0,
                "flow_score": 0,
                "character_score": 0,
                "clarity_score": 0,
                "genre_score": 0,
                "comments": f"Evaluation failed: {str(e)}",
                "evaluation_time": 0,
                "raw_response": ""
            }
    
    def parse_evaluation_response(self, response_text: str) -> Dict:     # Parse the AI evaluation response into structured scores
        
        scores = {
            "overall_score": 0,
            "flow_score": 0,
            "character_score": 0,
            "clarity_score": 0,
            "genre_score": 0,
            "comments": ""
        }
        
        # Extract scores using regex
        patterns = {
            "overall_score": r"OVERALL:\s*(\d+)%",
            "flow_score": r"FLOW:\s*(\d+)%",
            "character_score": r"CHARACTER:\s*(\d+)%",
            "clarity_score": r"CLARITY:\s*(\d+)%",
            "genre_score": r"GENRE:\s*(\d+)%"
        }
        
        for score_name, pattern in patterns.items():
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                scores[score_name] = int(match.group(1))
        
        # Extract comments
        comments_match = re.search(r"COMMENTS:\s*(.+?)(?:\n\n|$)", response_text, re.DOTALL | re.IGNORECASE)
        if comments_match:
            scores["comments"] = comments_match.group(1).strip()
        
        return scores
    
    def evaluate_chapter(self, chapter_num: int) -> EvaluationMetrics:       # Evaluate both baseline and enhanced translations for a chapter
        
        print(f"\nEvaluating Chapter {chapter_num}")
        print("-" * 50)
        
        try:
            # Load all translations
            baseline_text, enhanced_text, ground_truth = self.load_translations(chapter_num)
            print(f"Loaded translations - Baseline: {len(baseline_text)} chars, Enhanced: {len(enhanced_text)} chars")
            
            # Evaluate baseline translation
            print("Evaluating baseline translation...")
            baseline_eval = self.evaluate_translation(baseline_text, ground_truth, "baseline")
            
            # Evaluate enhanced translation  
            print("Evaluating enhanced translation...")
            enhanced_eval = self.evaluate_translation(enhanced_text, ground_truth, "enhanced")
            
            # Calculate improvement
            improvement = enhanced_eval["overall_score"] - baseline_eval["overall_score"]
            
            # Create metrics object
            metrics = EvaluationMetrics(
                chapter_num=chapter_num,
                baseline_score=baseline_eval["overall_score"],
                enhanced_score=enhanced_eval["overall_score"],
                improvement=improvement,
                flow_score=enhanced_eval["flow_score"],
                character_voice_score=enhanced_eval["character_score"],
                clarity_score=enhanced_eval["clarity_score"],
                genre_score=enhanced_eval["genre_score"],
                evaluator_comments=enhanced_eval["comments"],
                evaluation_time=baseline_eval["evaluation_time"] + enhanced_eval["evaluation_time"],
                timestamp=datetime.now().isoformat()
            )
            
            # Save detailed results
            self.save_chapter_evaluation(chapter_num, baseline_eval, enhanced_eval, metrics, 
                                       baseline_text, enhanced_text, ground_truth)
            
            self.evaluation_metrics.append(metrics)
            
            # Show results
            print(f"Results for Chapter {chapter_num}:")
            print(f"  Baseline Score: {baseline_eval['overall_score']}%")
            print(f"  Enhanced Score: {enhanced_eval['overall_score']}%") 
            print(f"  Improvement: {improvement:+.1f} points")
            print(f"  Enhanced Comments: {enhanced_eval['comments'][:100]}...")
            
            return metrics
            
        except Exception as e:
            print(f"Error evaluating chapter {chapter_num}: {e}")
            return None
    
    def save_chapter_evaluation(self, chapter_num: int, baseline_eval: Dict, enhanced_eval: Dict, 
                               metrics: EvaluationMetrics, baseline_text: str, enhanced_text: str, ground_truth: str):      # Save detailed evaluation results for a chapter
        
        # Save scoring results
        scores_file = Path(self.config.evaluation_output_dir, "scores", f"chapter_{chapter_num:04d}_scores.json")
        scores_data = {
            "chapter": chapter_num,
            "professional_baseline": 100.0,
            "baseline_evaluation": baseline_eval,
            "enhanced_evaluation": enhanced_eval,
            "improvement": enhanced_eval["overall_score"] - baseline_eval["overall_score"],
            "metrics": asdict(metrics),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(scores_file, 'w', encoding='utf-8') as f:
            json.dump(scores_data, f, indent=2, ensure_ascii=False)
        
        # Save side-by-side comparison for human review
        comparison_file = Path(self.config.evaluation_output_dir, "comparisons", f"chapter_{chapter_num:04d}_side_by_side.txt")
        with open(comparison_file, 'w', encoding='utf-8') as f:
            f.write(f"CHAPTER {chapter_num} TRANSLATION COMPARISON\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("PROFESSIONAL REFERENCE (100% baseline):\n")
            f.write("-" * 40 + "\n")
            f.write(ground_truth[:500] + "...\n\n")
            
            f.write(f"BASELINE TRANSLATION ({baseline_eval['overall_score']}%):\n")
            f.write("-" * 40 + "\n") 
            f.write(baseline_text[:500] + "...\n\n")
            
            f.write(f"ENHANCED TRANSLATION ({enhanced_eval['overall_score']}%):\n")
            f.write("-" * 40 + "\n")
            f.write(enhanced_text[:500] + "...\n\n")
            
            f.write("EVALUATION RESULTS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Improvement: {enhanced_eval['overall_score'] - baseline_eval['overall_score']:+.1f} points\n")
            f.write(f"Enhanced Comments: {enhanced_eval['comments']}\n")
    
    def run_evaluation_pipeline(self):      # Main evaluation pipeline
        
        print("Starting Translation Evaluation Pipeline")
        print(f"Evaluator Model: {self.config.evaluator_model}")
        print(f"Evaluating chapters {self.config.start_chapter}-{self.config.end_chapter}")
        print(f"Professional baseline = 100%")
        
        start_time = time.time()
        successful_evaluations = 0
        
        for chapter_num in range(self.config.start_chapter, self.config.end_chapter + 1):
            try:
                print(f"\n{'='*60}")
                metrics = self.evaluate_chapter(chapter_num)
                if metrics:
                    successful_evaluations += 1
                    print(f"Chapter {chapter_num} evaluated successfully")
                else:
                    print(f"Chapter {chapter_num} evaluation failed")
            except Exception as e:
                print(f"Failed to evaluate chapter {chapter_num}: {e}")
                continue
        
        # Final summary and analytics
        total_time = time.time() - start_time
        self.save_final_evaluation_report()
        
        print(f"\n{'='*60}")
        print("EVALUATION PIPELINE COMPLETE")
        print(f"Total time: {total_time:.1f}s")
        print(f"Chapters evaluated: {successful_evaluations}/{self.config.end_chapter - self.config.start_chapter + 1}")
        
        if self.evaluation_metrics:
            avg_baseline = sum(m.baseline_score for m in self.evaluation_metrics) / len(self.evaluation_metrics)
            avg_enhanced = sum(m.enhanced_score for m in self.evaluation_metrics) / len(self.evaluation_metrics)
            avg_improvement = avg_enhanced - avg_baseline
            
            print(f"\nRESULTS SUMMARY:")
            print(f"Average Baseline Score: {avg_baseline:.1f}%")
            print(f"Average Enhanced Score: {avg_enhanced:.1f}%") 
            print(f"Average Improvement: {avg_improvement:+.1f} points")
            
            if avg_improvement > 0:
                print(f"Rule learning shows improvement. Translations improved by {avg_improvement:.1f} points on average")
            elif avg_improvement < -2:
                print(f"Enhanced translations performed {abs(avg_improvement):.1f} points worse than baseline")
            else:
                print(f"Minimal change detected. Rule learning needs refinement")
    
    def save_final_evaluation_report(self):     # Save comprehensive evaluation analytics and report
        
        if not self.evaluation_metrics:
            print("No evaluation metrics to save")
            return
        
        # Calculate comprehensive statistics
        baseline_scores = [m.baseline_score for m in self.evaluation_metrics]
        enhanced_scores = [m.enhanced_score for m in self.evaluation_metrics]
        improvements = [m.improvement for m in self.evaluation_metrics]
        
        analytics = {
            "evaluation_config": asdict(self.config),
            "summary_statistics": {
                "chapters_evaluated": len(self.evaluation_metrics),
                "evaluator_model": self.config.evaluator_model,
                "evaluation_date": datetime.now().isoformat(),
                
                "baseline_performance": {
                    "average_score": round(sum(baseline_scores) / len(baseline_scores), 2),
                    "min_score": min(baseline_scores),
                    "max_score": max(baseline_scores),
                    "score_range": max(baseline_scores) - min(baseline_scores)
                },
                
                "enhanced_performance": {
                    "average_score": round(sum(enhanced_scores) / len(enhanced_scores), 2),
                    "min_score": min(enhanced_scores),
                    "max_score": max(enhanced_scores),
                    "score_range": max(enhanced_scores) - min(enhanced_scores)
                },
                
                "improvement_analysis": {
                    "average_improvement": round(sum(improvements) / len(improvements), 2),
                    "positive_improvements": len([x for x in improvements if x > 0]),
                    "negative_improvements": len([x for x in improvements if x < 0]),
                    "no_change": len([x for x in improvements if x == 0]),
                    "max_improvement": max(improvements),
                    "max_decline": min(improvements)
                }
            },
            "detailed_metrics": [asdict(m) for m in self.evaluation_metrics],
            "evaluation_conclusion": self.generate_evaluation_conclusion(improvements)
        }
        
        # Save analytics JSON
        analytics_file = Path(self.config.evaluation_output_dir, "analytics", "evaluation_analytics.json")
        with open(analytics_file, 'w', encoding='utf-8') as f:
            json.dump(analytics, f, indent=2, ensure_ascii=False)
        
        # Save human-readable report
        report_file = Path(self.config.evaluation_output_dir, "reports", "evaluation_report.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("TRANSLATION EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Evaluator Model: {self.config.evaluator_model}\n")
            f.write(f"Chapters Evaluated: {len(self.evaluation_metrics)}\n\n")
            
            f.write("PERFORMANCE SUMMARY:\n")
            f.write("-" * 20 + "\n")
            avg_baseline = sum(baseline_scores) / len(baseline_scores)
            avg_enhanced = sum(enhanced_scores) / len(enhanced_scores)
            avg_improvement = sum(improvements) / len(improvements)
            
            f.write(f"Baseline Translation Average: {avg_baseline:.1f}%\n")
            f.write(f"Enhanced Translation Average: {avg_enhanced:.1f}%\n")
            f.write(f"Average Improvement: {avg_improvement:+.1f} points\n\n")
            
            f.write("CHAPTER-BY-CHAPTER RESULTS:\n")
            f.write("-" * 30 + "\n")
            for metrics in self.evaluation_metrics:
                f.write(f"Chapter {metrics.chapter_num:2d}: ")
                f.write(f"Baseline {metrics.baseline_score:2.0f}% → ")
                f.write(f"Enhanced {metrics.enhanced_score:2.0f}% ")
                f.write(f"({metrics.improvement:+.1f})\n")
            
            f.write(f"\nCONCLUSION:\n")
            f.write("-" * 15 + "\n")
            f.write(analytics["evaluation_conclusion"])
        
        print(f"Evaluation analytics saved to: {analytics_file}")
        print(f"Human-readable report saved to: {report_file}")
    
    def generate_evaluation_conclusion(self, improvements: List[float]) -> str:      # Generate conclusion about rule learning effectiveness
        
        avg_improvement = sum(improvements) / len(improvements)
        positive_count = len([x for x in improvements if x > 0])
        total_count = len(improvements)
        
        if avg_improvement > 5:
            return f"Rule learning is highly effective. Enhanced translations averaged {avg_improvement:.1f} points better than baseline, with {positive_count}/{total_count} chapters showing improvement. The rule extraction and application system is working well."
        
        elif avg_improvement > 2:
            return f"Rule learning is working. Enhanced translations showed {avg_improvement:.1f} point average improvement over baseline. {positive_count}/{total_count} chapters improved. Consider expanding the rule database for better results."
        
        elif avg_improvement > -1:
            return f"Rule learning shows minimal impact. Average change of {avg_improvement:.1f} points suggests rules may need refinement. {positive_count}/{total_count} chapters improved. Review rule quality and application logic."
        
        else:
            return f"Enhanced translations performed {abs(avg_improvement):.1f} points worse than baseline on average. Only {positive_count}/{total_count} chapters improved. Rule learning may be applying incorrect patterns. Review rule extraction process."

def main():     # Main execution function for evaluation pipeline
    
    config = EvaluationConfig(
        start_chapter=1,
        end_chapter=3,
        evaluator_model="deepseek-chat",  # Use DeepSeek for unbiased evaluation
        temperature=0.2  # Low temperature for consistent, objective scoring
    )
    
    # Check API key
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("Error: DEEPSEEK_API_KEY not found in environment")
        print("Please set your DeepSeek API key in .env file for evaluation")
        return
    
    # Check if required directories exist
    required_dirs = [
        config.baseline_results_dir,
        config.enhanced_results_dir, 
        config.ground_truth_dir
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print("Error: Required directories not found:")
        for dir_path in missing_dirs:
            print(f"  - {dir_path}")
        print("\nPlease run the translation pipelines first:")
        print("  1. python scripts/1_baseline_translate.py")
        print("  2. python scripts/4_enhanced_translate.py")
        return
    
    print("Evaluation Configuration:")
    print(f"  Evaluator Model: {config.evaluator_model}")
    print(f"  Temperature: {config.temperature}")
    print(f"  Chapters: {config.start_chapter}-{config.end_chapter}")
    print(f"  Baseline Results: {config.baseline_results_dir}")
    print(f"  Enhanced Results: {config.enhanced_results_dir}")
    print(f"  Ground Truth: {config.ground_truth_dir}")
    print(f"  Output Directory: {config.evaluation_output_dir}")
    
    # Initialize and run evaluation
    evaluator = TranslationEvaluator(config)
    evaluator.run_evaluation_pipeline()

if __name__ == "__main__":
    main()