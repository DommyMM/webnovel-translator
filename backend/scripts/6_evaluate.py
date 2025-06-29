import asyncio
import argparse
import time
import sys
import subprocess
from pathlib import Path

async def run_script_async(script_name, args_list):
    cmd = ["python", script_name] + args_list
    print(f"Starting: {' '.join(cmd)}")
    
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    stdout, stderr = await process.communicate()
    
    return {
        "script": script_name,
        "returncode": process.returncode,
        "stdout": stdout.decode() if stdout else "",
        "stderr": stderr.decode() if stderr else ""
    }

def run_script_sync(script_name, args_list):
    cmd = ["python", script_name] + args_list
    print(f"Starting: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=False,  # Show output in real-time
            text=True,
            check=True
        )
        return {
            "script": script_name,
            "returncode": 0,
            "success": True
        }
    except subprocess.CalledProcessError as e:
        return {
            "script": script_name,
            "returncode": e.returncode,
            "success": False,
            "error": str(e)
        }

async def run_naive_evaluation_pipeline(start_chapter: int, end_chapter: int, concurrent: int = 3):
    print("=" * 70)
    print("STEP 6: NAIVE EVALUATION PIPELINE")
    print("=" * 70)
    print(f"Chapters: {start_chapter}-{end_chapter}")
    print(f"Max concurrent: {concurrent}")
    print()
    print("This will:")
    print("  6a. Generate naive translations (minimal prompt)")
    print("  6b. Compare naive vs enhanced translations (step 5)")
    print()
    
    start_time = time.time()
    
    # Step 6a: Generate naive translations
    print("=" * 50)
    print("STEP 6a: NAIVE TRANSLATION")
    print("=" * 50)
    
    step_6a_args = [
        "--start", str(start_chapter),
        "--end", str(end_chapter),
        "--concurrent", str(concurrent)
    ]
    
    step_6a_result = await run_script_async("6a_naive_translate.py", step_6a_args)
    
    if step_6a_result["returncode"] != 0:
        print(f"Step 6a failed (exit code {step_6a_result['returncode']})")
        if step_6a_result["stderr"]:
            print(f"Error: {step_6a_result['stderr'][:500]}...")
        return False
    
    print("Step 6a completed successfully")
    
    # Step 6b: Compare naive vs enhanced
    print("\n" + "=" * 50)
    print("STEP 6b: COMPARISON")
    print("=" * 50)
    
    step_6b_args = [
        "--start", str(start_chapter),
        "--end", str(end_chapter)
    ]
    
    # Run step 6b synchronously since it's just file processing
    step_6b_result = run_script_sync("6b_compare_naive_enhanced.py", step_6b_args)
    
    if not step_6b_result.get("success"):
        print(f"Step 6b failed (exit code {step_6b_result['returncode']})")
        if step_6b_result.get("error"):
            print(f"Error: {step_6b_result['error']}")
        return False
    
    print("Step 6b completed successfully")
    
    # Final summary
    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    
    print("\n" + "=" * 70)
    print("NAIVE EVALUATION PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Both steps completed successfully")
    print(f"Total time: {minutes}m {seconds}s")
    print(f"Chapters processed: {start_chapter}-{end_chapter}")
    print()
    print("Results available in:")
    print(f"  Naive translations: ../results/naive/translations/")
    print(f"  Comparisons: ../results/comparison/")
    print(f"  Summary: ../results/comparison/comparison_summary.txt")
    print()
    print("Check the comparison files to see what your pipeline improved")
    
    return True

def check_prerequisites(start_chapter: int, end_chapter: int):
    
    missing_files = []
    
    # Check Chinese chapters (needed for 6a)
    for chapter in range(start_chapter, end_chapter + 1):
        chinese_file = f"../data/chapters/clean/chapter_{chapter:04d}_cn.txt"
        if not Path(chinese_file).exists():
            missing_files.append(f"Chinese: {chinese_file}")
    
    # Check enhanced translations (needed for 6b)
    for chapter in range(start_chapter, end_chapter + 1):
        enhanced_file = f"../results/final/translations/chapter_{chapter:04d}_final.txt"
        if not Path(enhanced_file).exists():
            missing_files.append(f"Enhanced: {enhanced_file}")
    
    if missing_files:
        print("❌ Prerequisites not met. Missing files:")
        for file in missing_files[:10]:
            print(f"     {file}")
        if len(missing_files) > 10:
            print(f"     ... and {len(missing_files) - 10} more")
        print()
        print("Please run:")
        print("  • Step 5 (final translation) for enhanced translations")
        print("  • Or check that Chinese chapter files exist")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Step 6: Naive Evaluation Pipeline (6a + 6b)")
    parser.add_argument("--start", type=int, default=1, help="Start chapter number")
    parser.add_argument("--end", type=int, default=3, help="End chapter number")
    parser.add_argument("--concurrent", type=int, default=3, help="Max concurrent requests for step 6a")
    
    args = parser.parse_args()
    
    # Validation
    if args.start > args.end:
        print("Error: Start chapter must be <= end chapter")
        sys.exit(1)
    
    if args.concurrent < 1 or args.concurrent > 10:
        print("Error: Concurrent requests must be between 1 and 10")
        sys.exit(1)
    
    # Check API key (needed for step 6a)
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("Error: DEEPSEEK_API_KEY not found in environment")
        print("Step 6a requires API access for naive translation")
        sys.exit(1)
    
    # Check prerequisites
    if not check_prerequisites(args.start, args.end):
        sys.exit(1)
    
    # Change to script directory
    script_dir = Path(__file__).parent
    import os
    os.chdir(script_dir)
    
    print("Step 6 Configuration:")
    print(f"  Chapters: {args.start}-{args.end}")
    print(f"  Max concurrent (6a): {args.concurrent}")
    print(f"  Model: deepseek-chat")
    print(f"  Naive prompt: 'Translate this Chinese text to English prose:'")
    print(f"  Comparison: Naive (6a) vs Enhanced (step 5)")
    print()
    
    # Run pipeline
    success = asyncio.run(run_naive_evaluation_pipeline(args.start, args.end, args.concurrent))
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()