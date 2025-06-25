import os
import sys
import subprocess
import argparse
import time
from pathlib import Path
from dotenv import load_dotenv

def run_command(cmd, step_name, cwd=None):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"STEP: {step_name}")
    print(f"COMMAND: {' '.join(cmd)}")
    print('='*60)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd, 
            cwd=cwd,
            check=True, 
            capture_output=False,  # Show output in real-time
            text=True
        )
        
        elapsed = time.time() - start_time
        print(f"\n✓ {step_name} completed in {elapsed:.1f}s")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ {step_name} failed after {elapsed:.1f}s")
        print(f"Error code: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠ {step_name} interrupted by user")
        return False

def check_environment():
    """Check required environment variables"""
    # Load .env file from project root
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ Loaded .env from {env_path}")
    else:
        print(f"⚠ No .env file found at {env_path}")
    
    required_vars = ["DEEPSEEK_API_KEY", "CEREBRAS_API_KEY"]
    missing = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing.append(var)
    
    if missing:
        print("✗ Missing required environment variables:")
        for var in missing:
            print(f"  - {var}")
        print("\nPlease set these in your .env file or environment")
        return False
    
    print("✓ Environment variables found")
    return True

def check_files(start_chapter, end_chapter):
    """Check if required input files exist"""
    base_path = Path("../data/chapters")
    
    # Check if base directories exist
    required_dirs = [
        base_path / "clean",
        base_path / "ground_truth"
    ]
    
    for dir_path in required_dirs:
        if not dir_path.exists():
            print(f"✗ Required directory not found: {dir_path}")
            return False
    
    # Check if chapter files exist
    missing_files = []
    for chapter in range(start_chapter, end_chapter + 1):
        chinese_file = base_path / "clean" / f"chapter_{chapter:04d}_cn.txt"
        english_file = base_path / "ground_truth" / f"chapter_{chapter:04d}_en.txt"
        
        if not chinese_file.exists():
            missing_files.append(str(chinese_file))
        if not english_file.exists():
            missing_files.append(str(english_file))
    
    if missing_files:
        print("✗ Missing required chapter files:")
        for file_path in missing_files[:5]:  # Show first 5
            print(f"  - {file_path}")
        if len(missing_files) > 5:
            print(f"  ... and {len(missing_files) - 5} more")
        return False
    
    print(f"✓ Chapter files found for chapters {start_chapter}-{end_chapter}")
    return True

def run_full_pipeline(start_chapter, end_chapter, concurrent=3):
    """Run the complete translation pipeline"""
    
    print("Translation Pipeline Runner")
    print(f"Chapters: {start_chapter}-{end_chapter}")
    print(f"Concurrent requests: {concurrent}")
    
    # Pre-flight checks
    print("\n" + "="*60)
    print("PRE-FLIGHT CHECKS")
    print("="*60)
    
    if not check_environment():
        return False
    
    if not check_files(start_chapter, end_chapter):
        return False
    
    # Pipeline steps
    steps = [
        {
            "name": "Step 1: Baseline Translation",
            "cmd": ["python", "1_baseline_translate.py"],
            "note": "⚠ Uses hardcoded config - edit script if needed"
        },
        {
            "name": "Step 2: Parallel Extraction (Rules + Terminology)", 
            "cmd": ["python", "2_parallel_extraction.py", "--start", str(start_chapter), "--end", str(end_chapter), "--concurrent", str(concurrent)]
        },
        {
            "name": "Step 3: Parallel Cleaning (Rules)",
            "cmd": ["python", "3_parallel_cleaning.py"]
        },
        {
            "name": "Step 4: Build ChromaDB",
            "cmd": ["python", "4_build_chromadb.py"]
        },
        {
            "name": "Step 5: Final Translation (Rules + RAG)",
            "cmd": ["python", "5_final_translate.py", "--start", str(start_chapter), "--end", str(end_chapter), "--concurrent", str(concurrent)]
        },
        {
            "name": "Step 6: Evaluation",
            "cmd": ["python", "6_evaluate.py", "--start", str(start_chapter), "--end", str(end_chapter), "--concurrent", str(concurrent)]
        }
    ]
    
    # Run pipeline
    print(f"\n{'='*60}")
    print(f"STARTING PIPELINE - {len(steps)} STEPS")
    print('='*60)
    
    pipeline_start = time.time()
    
    for i, step in enumerate(steps, 1):
        if "note" in step:
            print(f"\n{step['note']}")
        
        success = run_command(step["cmd"], f"{i}/6 - {step['name']}", cwd=".")
        
        if not success:
            print(f"\n✗ Pipeline failed at step {i}")
            print("Fix the error and re-run the pipeline")
            return False
    
    # Success
    pipeline_elapsed = time.time() - pipeline_start
    minutes = int(pipeline_elapsed // 60)
    seconds = int(pipeline_elapsed % 60)
    
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print('='*60)
    print(f"✓ All 6 steps completed successfully")
    print(f"✓ Total time: {minutes}m {seconds}s")
    print(f"✓ Chapters processed: {start_chapter}-{end_chapter}")
    print(f"\nResults available in:")
    print(f"  - ../results/baseline/translations/")
    print(f"  - ../results/final/translations/")
    print(f"\nEvaluation completed automatically in step 6")
    print(f"Check: ../results/evaluation/reports/evaluation_report.txt")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Run complete translation pipeline")
    parser.add_argument("--start", type=int, default=1, help="Start chapter number")
    parser.add_argument("--end", type=int, default=3, help="End chapter number") 
    parser.add_argument("--concurrent", type=int, default=3, help="Max concurrent requests")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.start > args.end:
        print("Error: Start chapter must be <= end chapter")
        return
    
    if args.concurrent < 1 or args.concurrent > 10:
        print("Error: Concurrent requests must be between 1 and 10")
        return
    
    # Change to scripts directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print(f"Working directory: {os.getcwd()}")
    
    # Run pipeline
    success = run_full_pipeline(args.start, args.end, args.concurrent)
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()