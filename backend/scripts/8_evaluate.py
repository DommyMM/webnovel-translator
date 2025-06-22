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
class CompleteEvaluationMetrics:
    chapter_num: int
    baseline_score: float
    enhanced_score: float
    final_score: float
    professional_score: float = 100.0
    rules_improvement: float = 0.0  # enhanced - baseline
    rag_improvement: float = 0.0    # final - enhanced
    total_improvement: float = 0.0  # final - baseline
    flow_score: float = 0.0
    character_voice_score: float = 0.0
    clarity_score: float = 0.0
    genre_score: float = 0.0
    evaluator_comments: str = ""
    evaluation_time: float = 0.0
    timestamp: str = ""

@dataclass
class CompleteEvaluationConfig:
    baseline_results_dir: str = "../results/baseline"
    enhanced_results_dir: str = "../results/enhanced" 
    final_results_dir: str = "../results/final"
    ground_truth_dir: str = "../data/chapters/ground_truth"
    evaluation_output_dir: str = "../results/evaluation_complete"
    start_chapter: int = 1
    end_chapter: int = 3
    evaluator_model: str = "deepseek-chat"
    temperature: float = 0.2  # Low temp for consistent scoring
    max_tokens: int = 8192
    base_url: str = "https://api.deepseek.com"
    max_concurrent: int = 10

class AsyncCompleteEvaluator:
    def __init__(self, config: CompleteEvaluationConfig, chapter_num: int):
        self.config = config
        self.chapter_num = chapter_num
        self.client = AsyncOpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=self.config.base_url
        )
        self.setup_directories()
        
    def setup_directories(self):
        Path(self.config.evaluation_output_dir).mkdir(exist_ok=True)
        for subdir in ["scores", "comparisons", "analytics", "reports"]:
            Path(self.config.evaluation_output_dir, subdir).mkdir(exist_ok=True)
    
    def load_all_translations(self) -> tuple[str, str, str, str]:
        """Load baseline, enhanced, final, and ground truth translations"""
        
        # Load baseline translation
        baseline_file = Path(self.config.baseline_results_dir, "translations", f"chapter_{self.chapter_num:04d}_deepseek.txt")
        if not baseline_file.exists():
            raise FileNotFoundError(f"Baseline translation not found: {baseline_file}")
        
        with open(baseline_file, 'r', encoding='utf-8') as f:
            baseline_text = f.read().strip()
            if baseline_text.startswith("DeepSeek Translation"):
                lines = baseline_text.split('\n')
                baseline_text = '\n'.join(lines[2:]).strip()
        
        # Load enhanced translation (rules only)
        enhanced_file = Path(self.config.enhanced_results_dir, "translations", f"chapter_{self.chapter_num:04d}_enhanced.txt")
        if not enhanced_file.exists():
            raise FileNotFoundError(f"Enhanced translation not found: {enhanced_file}")
        
        with open(enhanced_file, 'r', encoding='utf-8') as f:
            enhanced_text = f.read().strip()
            if enhanced_text.startswith("Enhanced Translation"):
                lines = enhanced_text.split('\n')
                enhanced_text = '\n'.join(lines[3:]).strip()
        
        # Load final translation (rules + RAG)
        final_file = Path(self.config.final_results_dir, "translations", f"chapter_{self.chapter_num:04d}_final.txt")
        if not final_file.exists():
            raise FileNotFoundError(f"Final translation not found: {final_file}")
        
        with open(final_file, 'r', encoding='utf-8') as f:
            final_text = f.read().strip()
            if final_text.startswith("Final Translation"):
                lines = final_text.split('\n')
                final_text = '\n'.join(lines[5:]).strip()  # Skip more header lines
        
        # Load professional ground truth
        truth_file = Path(self.config.ground_truth_dir, f"chapter_{self.chapter_num:04d}_en.txt")
        if not truth_file.exists():
            raise FileNotFoundError(f"Ground truth not found: {truth_file}")
        
        with open(truth_file, 'r', encoding='utf-8') as f:
            ground_truth = f.read().strip()
            lines = ground_truth.split('\n')
            if lines[0].startswith("Chapter"):
                ground_truth = '\n'.join(lines[3:]).strip()
        
        return baseline_text, enhanced_text, final_text, ground_truth
    
    def create_evaluation_prompt(self, translation: str, ground_truth: str, translation_type: str) -> str:
        """Create evaluation prompt for scoring translation quality"""
        
        prompt = f"""You are a Western reader who enjoys cultivation novels. Your job is to rate how much you'd enjoy reading this translation compared to the professional version.

PROFESSIONAL REFERENCE (your 100% gold standard):
{ground_truth[:800]}...

{translation_type.upper()} TRANSLATION (rate this version):
{translation[:800]}...

Your Task: Rate this translation as a Western reader who wants to enjoy the story.

What You Care About (as a Western cultivation novel reader):
- Natural English Flow: Does it read smoothly like a real English novel?
- Character Personality: Do characters feel real and consistent?  
- Story Enjoyment: Can you follow the action and get invested?
- Proper Cultivation Terms: Do terms like "Spiritual Strength" feel right?
- Western Style: Written for Western readers colloquially, not literal translation

Scoring Scale:
- 95-100%: Perfect - reads like a professional English novel
- 85-95%: Excellent - reads like professional English, would prefer this over many fan translations
- 70-84%: Very good - smooth reading with minor artifacts, fully enjoyable
- 55-69%: Good - readable with some awkward phrasing, but story flows well
- 40-54%: Acceptable - clearly machine translated but understandable
- 25-39%: Poor - heavy machine translation artifacts, difficult to follow story
- 10-24%: Very poor - broken English, major comprehension issues
- 0-9%: Unreadable - incomprehensible, completely failed translation

Rate these aspects:
1. Overall Enjoyment: How much would you enjoy this vs professional?
2. Reading Flow: Does it read smoothly?
3. Character Voice: Do characters feel real and consistent?
4. Story Clarity: Can you follow what's happening?
5. Genre Feel: Does it feel like a proper cultivation novel?

Response Format:
OVERALL: [score]%
FLOW: [score]%
CHARACTER: [score]%  
CLARITY: [score]%
GENRE: [score]%
COMMENTS: [2-3 sentences about what works well or needs improvement for reader enjoyment]"""
        
        return prompt
    
    async def evaluate_translation_async(self, translation: str, ground_truth: str, translation_type: str) -> Dict:
        """Evaluate a single translation against ground truth"""
        
        start_time = time.time()
        prompt = self.create_evaluation_prompt(translation, ground_truth, translation_type)
        
        try:
            response = await self.client.chat.completions.create(
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
            print(f"Chapter {self.chapter_num}: Evaluation failed for {translation_type}: {str(e)}")
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
    
    def parse_evaluation_response(self, response_text: str) -> Dict:
        """Parse evaluation response to extract scores"""
        
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
    
    async def evaluate_chapter_complete_async(self, semaphore: asyncio.Semaphore) -> Optional[CompleteEvaluationMetrics]:
        """Evaluate all three translation versions for a chapter"""
        
        async with semaphore:
            print(f"Starting Chapter {self.chapter_num} complete evaluation...")
            
            try:
                # Load all translations
                baseline_text, enhanced_text, final_text, ground_truth = self.load_all_translations()
                print(f"Chapter {self.chapter_num}: Loaded all translations")
                
                # Evaluate all three versions
                print(f"Chapter {self.chapter_num}: Evaluating baseline...")
                baseline_eval = await self.evaluate_translation_async(baseline_text, ground_truth, "baseline")
                
                print(f"Chapter {self.chapter_num}: Evaluating enhanced...")
                enhanced_eval = await self.evaluate_translation_async(enhanced_text, ground_truth, "enhanced")
                
                print(f"Chapter {self.chapter_num}: Evaluating final...")
                final_eval = await self.evaluate_translation_async(final_text, ground_truth, "final")
                
                # Calculate improvements
                rules_improvement = enhanced_eval["overall_score"] - baseline_eval["overall_score"]
                rag_improvement = final_eval["overall_score"] - enhanced_eval["overall_score"]
                total_improvement = final_eval["overall_score"] - baseline_eval["overall_score"]
                
                # Create metrics object
                metrics = CompleteEvaluationMetrics(
                    chapter_num=self.chapter_num,
                    baseline_score=baseline_eval["overall_score"],
                    enhanced_score=enhanced_eval["overall_score"],
                    final_score=final_eval["overall_score"],
                    rules_improvement=rules_improvement,
                    rag_improvement=rag_improvement,
                    total_improvement=total_improvement,
                    flow_score=final_eval["flow_score"],
                    character_voice_score=final_eval["character_score"],
                    clarity_score=final_eval["clarity_score"],
                    genre_score=final_eval["genre_score"],
                    evaluator_comments=final_eval["comments"],
                    evaluation_time=baseline_eval["evaluation_time"] + enhanced_eval["evaluation_time"] + final_eval["evaluation_time"],
                    timestamp=datetime.now().isoformat()
                )
                
                # Save detailed results
                self.save_complete_evaluation(baseline_eval, enhanced_eval, final_eval, metrics,
                                            baseline_text, enhanced_text, final_text, ground_truth)
                
                # Show completion
                print(f"Chapter {self.chapter_num} complete:")
                print(f"  Baseline: {baseline_eval['overall_score']}%")
                print(f"  Enhanced: {enhanced_eval['overall_score']}% ({rules_improvement:+.1f})")
                print(f"  Final: {final_eval['overall_score']}% ({rag_improvement:+.1f})")
                print(f"  Total improvement: {total_improvement:+.1f}")
                
                return metrics
                
            except Exception as e:
                print(f"Chapter {self.chapter_num}: Error evaluating: {e}")
                return None
    
    def save_complete_evaluation(self, baseline_eval: Dict, enhanced_eval: Dict, final_eval: Dict,
                                metrics: CompleteEvaluationMetrics, baseline_text: str, enhanced_text: str,
                                final_text: str, ground_truth: str):
        """Save detailed evaluation results"""
        
        # Save scoring results
        scores_file = Path(self.config.evaluation_output_dir, "scores", f"chapter_{self.chapter_num:04d}_complete_scores.json")
        scores_data = {
            "chapter": self.chapter_num,
            "professional_baseline": 100.0,
            "baseline_evaluation": baseline_eval,
            "enhanced_evaluation": enhanced_eval,
            "final_evaluation": final_eval,
            "improvements": {
                "rules_improvement": enhanced_eval["overall_score"] - baseline_eval["overall_score"],
                "rag_improvement": final_eval["overall_score"] - enhanced_eval["overall_score"],
                "total_improvement": final_eval["overall_score"] - baseline_eval["overall_score"]
            },
            "metrics": asdict(metrics),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(scores_file, 'w', encoding='utf-8') as f:
            json.dump(scores_data, f, indent=2, ensure_ascii=False)
        
        # Save side-by-side comparison
        comparison_file = Path(self.config.evaluation_output_dir, "comparisons", f"chapter_{self.chapter_num:04d}_complete_comparison.txt")
        with open(comparison_file, 'w', encoding='utf-8') as f:
            f.write(f"CHAPTER {self.chapter_num} COMPLETE TRANSLATION COMPARISON\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("PROFESSIONAL REFERENCE (100% baseline):\n")
            f.write("-" * 50 + "\n")
            f.write(ground_truth[:400] + "...\n\n")
            
            f.write(f"BASELINE TRANSLATION ({baseline_eval['overall_score']}%):\n")
            f.write("-" * 50 + "\n") 
            f.write(baseline_text[:400] + "...\n\n")
            
            f.write(f"ENHANCED TRANSLATION - Rules Only ({enhanced_eval['overall_score']}%):\n")
            f.write("-" * 50 + "\n")
            f.write(enhanced_text[:400] + "...\n\n")
            
            f.write(f"FINAL TRANSLATION - Rules + RAG ({final_eval['overall_score']}%):\n")
            f.write("-" * 50 + "\n")
            f.write(final_text[:400] + "...\n\n")
            
            f.write("IMPROVEMENT ANALYSIS:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Rules improvement (Enhanced vs Baseline): {enhanced_eval['overall_score'] - baseline_eval['overall_score']:+.1f} points\n")
            f.write(f"RAG improvement (Final vs Enhanced): {final_eval['overall_score'] - enhanced_eval['overall_score']:+.1f} points\n") 
            f.write(f"Total improvement (Final vs Baseline): {final_eval['overall_score'] - baseline_eval['overall_score']:+.1f} points\n\n")
            f.write(f"Final Comments: {final_eval['comments']}\n")

class AsyncCompleteEvaluationPipeline:
    def __init__(self, config: CompleteEvaluationConfig):
        self.config = config
        self.setup_directories()
        
    def setup_directories(self):
        Path(self.config.evaluation_output_dir).mkdir(exist_ok=True)
        for subdir in ["scores", "comparisons", "analytics", "reports"]:
            Path(self.config.evaluation_output_dir, subdir).mkdir(exist_ok=True)
    
    async def run_async_complete_evaluation(self):
        """Run complete evaluation pipeline on all chapters"""
        
        print("Starting Complete Translation Evaluation Pipeline")
        print(f"Evaluator Model: {self.config.evaluator_model}")
        print(f"Max concurrent requests: {self.config.max_concurrent}")
        print(f"Evaluating chapters {self.config.start_chapter}-{self.config.end_chapter}")
        print(f"Comparing: Baseline vs Enhanced vs Final vs Professional")
        
        start_time = time.time()
        chapters = list(range(self.config.start_chapter, self.config.end_chapter + 1))
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        # Create evaluators for each chapter
        evaluators = [AsyncCompleteEvaluator(self.config, chapter_num) for chapter_num in chapters]
        
        # Create tasks for all chapters
        tasks = [evaluator.evaluate_chapter_complete_async(semaphore) for evaluator in evaluators]
        
        print(f"Launching {len(tasks)} concurrent evaluation tasks...")
        
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
        
        evaluation_time = time.time() - start_time
        print(f"Complete evaluation finished in {evaluation_time:.1f}s")
        print(f"Successfully evaluated {len(successful_metrics)}/{len(chapters)} chapters")
        
        if failed_count > 0:
            print(f"Failed chapters: {failed_count}")
        
        # Save analytics and generate report
        print("Generating final evaluation report...")
        self.save_complete_evaluation_report(successful_metrics)
        
        # Show final summary
        if successful_metrics:
            avg_baseline = sum(m.baseline_score for m in successful_metrics) / len(successful_metrics)
            avg_enhanced = sum(m.enhanced_score for m in successful_metrics) / len(successful_metrics)
            avg_final = sum(m.final_score for m in successful_metrics) / len(successful_metrics)
            avg_rules_improvement = sum(m.rules_improvement for m in successful_metrics) / len(successful_metrics)
            avg_rag_improvement = sum(m.rag_improvement for m in successful_metrics) / len(successful_metrics)
            avg_total_improvement = sum(m.total_improvement for m in successful_metrics) / len(successful_metrics)
            
            print(f"\nCOMPLETE PIPELINE RESULTS:")
            print("=" * 60)
            print(f"Average Baseline Score: {avg_baseline:.1f}%")
            print(f"Average Enhanced Score: {avg_enhanced:.1f}% ({avg_rules_improvement:+.1f} from rules)")
            print(f"Average Final Score: {avg_final:.1f}% ({avg_rag_improvement:+.1f} from RAG)")
            print(f"Total Average Improvement: {avg_total_improvement:+.1f} points")
            
            print(f"\nIMPACT BREAKDOWN:")
            print(f"  Rules Impact: {avg_rules_improvement:+.1f} points")
            print(f"  RAG Impact: {avg_rag_improvement:+.1f} points")
            print(f"  Combined Impact: {avg_total_improvement:+.1f} points")
            
            # Simple assessment without emojis
            if avg_total_improvement > 5:
                print(f"\nRESULT: Pipeline shows significant improvement")
            elif avg_total_improvement > 2:
                print(f"\nRESULT: Pipeline shows meaningful improvement")
            else:
                print(f"\nRESULT: Pipeline improvement is limited")
        
        return successful_metrics
    
    def save_complete_evaluation_report(self, metrics: List[CompleteEvaluationMetrics]):
        """Save comprehensive evaluation analytics and report"""
        
        if not metrics:
            print("No evaluation metrics to save")
            return
        
        # Calculate comprehensive statistics
        baseline_scores = [m.baseline_score for m in metrics]
        enhanced_scores = [m.enhanced_score for m in metrics]
        final_scores = [m.final_score for m in metrics]
        rules_improvements = [m.rules_improvement for m in metrics]
        rag_improvements = [m.rag_improvement for m in metrics]
        total_improvements = [m.total_improvement for m in metrics]
        
        analytics = {
            "evaluation_config": asdict(self.config),
            "summary_statistics": {
                "chapters_evaluated": len(metrics),
                "evaluator_model": self.config.evaluator_model,
                "evaluation_date": datetime.now().isoformat(),
                
                "baseline_performance": {
                    "average_score": round(sum(baseline_scores) / len(baseline_scores), 2),
                    "min_score": min(baseline_scores),
                    "max_score": max(baseline_scores)
                },
                
                "enhanced_performance": {
                    "average_score": round(sum(enhanced_scores) / len(enhanced_scores), 2),
                    "min_score": min(enhanced_scores),
                    "max_score": max(enhanced_scores)
                },
                
                "final_performance": {
                    "average_score": round(sum(final_scores) / len(final_scores), 2),
                    "min_score": min(final_scores),
                    "max_score": max(final_scores)
                },
                
                "improvement_analysis": {
                    "avg_rules_improvement": round(sum(rules_improvements) / len(rules_improvements), 2),
                    "avg_rag_improvement": round(sum(rag_improvements) / len(rag_improvements), 2),
                    "avg_total_improvement": round(sum(total_improvements) / len(total_improvements), 2),
                    "positive_rules_impact": len([x for x in rules_improvements if x > 0]),
                    "positive_rag_impact": len([x for x in rag_improvements if x > 0]),
                    "positive_total_impact": len([x for x in total_improvements if x > 0])
                }
            },
            "detailed_metrics": [asdict(m) for m in metrics],
            "evaluation_conclusion": self.generate_pipeline_conclusion(total_improvements, rules_improvements, rag_improvements)
        }
        
        # Save analytics JSON
        analytics_file = Path(self.config.evaluation_output_dir, "analytics", "complete_evaluation_analytics.json")
        with open(analytics_file, 'w', encoding='utf-8') as f:
            json.dump(analytics, f, indent=2, ensure_ascii=False)
        
        # Save human-readable report
        report_file = Path(self.config.evaluation_output_dir, "reports", "complete_evaluation_report.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("COMPLETE TRANSLATION PIPELINE EVALUATION REPORT\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Evaluator Model: {self.config.evaluator_model}\n")
            f.write(f"Chapters Evaluated: {len(metrics)}\n\n")
            
            f.write("PIPELINE PERFORMANCE PROGRESSION:\n")
            f.write("-" * 50 + "\n")
            avg_baseline = sum(baseline_scores) / len(baseline_scores)
            avg_enhanced = sum(enhanced_scores) / len(enhanced_scores)
            avg_final = sum(final_scores) / len(final_scores)
            avg_rules_improvement = sum(rules_improvements) / len(rules_improvements)
            avg_rag_improvement = sum(rag_improvements) / len(rag_improvements)
            avg_total_improvement = sum(total_improvements) / len(total_improvements)
            
            f.write(f"1. Baseline Translation: {avg_baseline:.1f}%\n")
            f.write(f"2. Enhanced (+ Rules): {avg_enhanced:.1f}% ({avg_rules_improvement:+.1f})\n")
            f.write(f"3. Final (+ Rules + RAG): {avg_final:.1f}% ({avg_rag_improvement:+.1f})\n")
            f.write(f"Total Pipeline Improvement: {avg_total_improvement:+.1f} points\n\n")
            
            f.write("IMPACT BREAKDOWN:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Rules Impact: {avg_rules_improvement:+.1f} points\n")
            f.write(f"RAG Impact: {avg_rag_improvement:+.1f} points\n")
            f.write(f"Combined Impact: {avg_total_improvement:+.1f} points\n\n")
            
            f.write("CHAPTER-BY-CHAPTER RESULTS:\n")
            f.write("-" * 40 + "\n")
            f.write("Ch# | Baseline | Enhanced | Final | Rules | RAG | Total\n")
            f.write("-" * 60 + "\n")
            for metric in metrics:
                f.write(f"{metric.chapter_num:2d}  | ")
                f.write(f"{metric.baseline_score:6.1f}%  | ")
                f.write(f"{metric.enhanced_score:6.1f}%  | ")
                f.write(f"{metric.final_score:5.1f}% | ")
                f.write(f"{metric.rules_improvement:+4.1f} | ")
                f.write(f"{metric.rag_improvement:+4.1f} | ")
                f.write(f"{metric.total_improvement:+4.1f}\n")
            
            f.write(f"\nCONCLUSION:\n")
            f.write("-" * 15 + "\n")
            f.write(analytics["evaluation_conclusion"])
        
        print(f"Complete evaluation analytics saved to: {analytics_file}")
        print(f"Human-readable report saved to: {report_file}")
    
    def generate_pipeline_conclusion(self, total_improvements: List[float], rules_improvements: List[float], rag_improvements: List[float]) -> str:
        """Generate objective conclusion about pipeline effectiveness"""
        
        avg_total = sum(total_improvements) / len(total_improvements)
        avg_rules = sum(rules_improvements) / len(rules_improvements)
        avg_rag = sum(rag_improvements) / len(rag_improvements)
        
        positive_total = len([x for x in total_improvements if x > 0])
        total_count = len(total_improvements)
        
        if avg_total > 10:
            return f"The translation pipeline is highly effective with {avg_total:.1f} point average improvement. Both rules ({avg_rules:+.1f}) and RAG ({avg_rag:+.1f}) contribute significantly. {positive_total}/{total_count} chapters improved. This represents professional-grade enhancement."
        
        elif avg_total > 5:
            return f"The translation pipeline shows strong effectiveness with {avg_total:.1f} point average improvement. Rules contribute {avg_rules:+.1f} points and RAG adds {avg_rag:+.1f} points. {positive_total}/{total_count} chapters improved. The system successfully enhances translation quality."
        
        elif avg_total > 2:
            return f"The translation pipeline shows meaningful improvement with {avg_total:.1f} point average gain. Rules ({avg_rules:+.1f}) and RAG ({avg_rag:+.1f}) both contribute positively. {positive_total}/{total_count} chapters improved. Consider expanding the rule and terminology databases for greater impact."
        
        elif avg_total > 0:
            return f"The translation pipeline shows modest improvement with {avg_total:.1f} point average gain. Rules contribute {avg_rules:+.1f} and RAG {avg_rag:+.1f}. {positive_total}/{total_count} chapters improved. The approach is promising but needs refinement for stronger impact."
        
        else:
            return f"The translation pipeline shows limited effectiveness with {avg_total:.1f} average change. Rules ({avg_rules:+.1f}) and RAG ({avg_rag:+.1f}) impacts vary. Only {positive_total}/{total_count} chapters improved. Consider reviewing rule extraction and terminology quality."

def main():
    """Main evaluation pipeline entry point"""
    
    parser = argparse.ArgumentParser(description="Complete Translation Evaluation Pipeline")
    parser.add_argument("--chapter", type=int, help="Evaluate single chapter")
    parser.add_argument("--start", type=int, default=1, help="Start chapter")
    parser.add_argument("--end", type=int, default=3, help="End chapter")
    parser.add_argument("--concurrent", type=int, default=3, help="Max concurrent requests")
    
    args = parser.parse_args()
    
    config = CompleteEvaluationConfig(
        start_chapter=args.start,
        end_chapter=args.end,
        max_concurrent=args.concurrent
    )
    
    # Check API key
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("Error: DEEPSEEK_API_KEY not found in environment")
        return
    
    # Check if required directories exist
    required_dirs = [
        config.baseline_results_dir,
        config.enhanced_results_dir, 
        config.final_results_dir,
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
        print("\nPlease run the complete pipeline first:")
        print("  1. python 1_baseline_translate.py")
        print("  2-3. python 2_extract_rules.py && python 3_clean_rules.py")
        print("  4. python 4_enhanced_translate.py")
        print("  5-6. python 5_extract_terminology.py && python 6_clean_terms.py")
        print("  7. python 7_final_translate.py")
        return
    
    if args.chapter:
        # Single chapter mode
        print(f"Evaluating single chapter: {args.chapter}")
        async def single_chapter():
            semaphore = asyncio.Semaphore(1)
            evaluator = AsyncCompleteEvaluator(config, args.chapter)
            result = await evaluator.evaluate_chapter_complete_async(semaphore)
            if result:
                print(f"Chapter {args.chapter} evaluation complete")
                print(f"  Baseline: {result.baseline_score}%")
                print(f"  Enhanced: {result.enhanced_score}% ({result.rules_improvement:+.1f})")
                print(f"  Final: {result.final_score}% ({result.rag_improvement:+.1f})")
                print(f"  Total improvement: {result.total_improvement:+.1f}")
            else:
                print(f"Chapter {args.chapter} evaluation failed")
        
        asyncio.run(single_chapter())
    else:
        # Evaluate all chapters
        print("Complete Evaluation Configuration:")
        print(f"  Evaluator Model: {config.evaluator_model}")
        print(f"  Temperature: {config.temperature}")
        print(f"  Max concurrent: {config.max_concurrent}")
        print(f"  Chapters: {config.start_chapter}-{config.end_chapter}")
        print(f"  Baseline Results: {config.baseline_results_dir}")
        print(f"  Enhanced Results: {config.enhanced_results_dir}")
        print(f"  Final Results: {config.final_results_dir}")
        print(f"  Output Directory: {config.evaluation_output_dir}")
        
        pipeline = AsyncCompleteEvaluationPipeline(config)
        asyncio.run(pipeline.run_async_complete_evaluation())

if __name__ == "__main__":
    main()