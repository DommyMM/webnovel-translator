import asyncio
import argparse
import time
import sys
from pathlib import Path

async def run_script_async(script_name, args_list):
    """Run a Python script asynchronously with given arguments"""
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

async def run_parallel_extraction(start_chapter, end_chapter, concurrent=3):
    print("=" * 70)
    print("PARALLEL EXTRACTION - Steps 2a & 2b")
    print("=" * 70)
    print(f"Chapters: {start_chapter}-{end_chapter}")
    print(f"Concurrent requests: {concurrent}")
    print()
    
    # Prepare arguments for both scripts
    common_args = [
        "--start", str(start_chapter),
        "--end", str(end_chapter), 
        "--concurrent", str(concurrent)
    ]
    
    start_time = time.time()
    
    # Run both extractions in parallel
    print("Starting parallel extraction...")
    print("  2a: Rule extraction (baseline vs professional)")
    print("  2b: Terminology extraction (baseline vs professional)")
    print()
    
    tasks = [
        run_script_async("2a_extract_rules.py", common_args),
        run_script_async("2b_extract_terminology.py", common_args)
    ]
    
    # Wait for both to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    
    print("=" * 70)
    print("PARALLEL EXTRACTION RESULTS")
    print("=" * 70)
    
    success_count = 0
    
    for result in results:
        if isinstance(result, Exception):
            print(f"Exception occurred: {result}")
            continue
            
        script = result["script"]
        returncode = result["returncode"]
        
        if returncode == 0:
            print(f"{script}: SUCCESS")
            success_count += 1
        else:
            print(f"{script}: FAILED (exit code {returncode})")
            if result["stderr"]:
                print(f"   Error: {result['stderr'][:200]}...")
    
    print(f"\nTotal time: {minutes}m {seconds}s")
    print(f"Success rate: {success_count}/2 scripts")
    
    if success_count == 2:
        print("\nAll extractions completed successfully.")
        print("Next steps:")
        print("   python 3_parallel_cleaning.py")
        return True
    else:
        print("\nSome extractions failed. Check logs above.")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run parallel extraction (steps 2a & 2b)")
    parser.add_argument("--start", type=int, default=1, help="Start chapter number")
    parser.add_argument("--end", type=int, default=3, help="End chapter number")
    parser.add_argument("--concurrent", type=int, default=3, help="Max concurrent requests per script")
    
    args = parser.parse_args()
    
    # Validation
    if args.start > args.end:
        print("❌ Error: Start chapter must be <= end chapter")
        sys.exit(1)
    
    if args.concurrent < 1 or args.concurrent > 10:
        print("❌ Error: Concurrent requests must be between 1 and 10") 
        sys.exit(1)
    
    # Change to script directory
    script_dir = Path(__file__).parent
    import os
    os.chdir(script_dir)
    
    # Run parallel extraction
    success = asyncio.run(run_parallel_extraction(args.start, args.end, args.concurrent))
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
