import asyncio
import argparse
import time
import sys
from pathlib import Path

async def run_script_async(script_name, args_list=None):
    cmd = ["python", script_name]
    if args_list:
        cmd.extend(args_list)
    
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

async def run_parallel_cleaning():
    print("=" * 70)
    print("PARALLEL CLEANING - Steps 3a & 3b")
    print("=" * 70)
    print()
    
    start_time = time.time()
    
    # Run both cleaning steps in parallel
    print("🧹 Starting parallel cleaning...")
    print("  📝 3a: Rule cleaning (Cerebras AI refinement)")
    print("  📚 3b: Terminology cleaning (ChromaDB preparation - now in step 4)")
    print()
    
    tasks = [
        run_script_async("3a_clean_rules.py"),
        # Note: 3b is now moved to step 4, so we don't run it here
    ]
    
    # Wait for cleaning to complete (just rules for now)
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    
    print("=" * 70)
    print("PARALLEL CLEANING RESULTS")
    print("=" * 70)
    
    success_count = 0
    
    for result in results:
        if isinstance(result, Exception):
            print(f"❌ Exception occurred: {result}")
            continue
            
        script = result["script"]
        returncode = result["returncode"]
        
        if returncode == 0:
            print(f"✅ {script}: SUCCESS")
            success_count += 1
        else:
            print(f"❌ {script}: FAILED (exit code {returncode})")
            if result["stderr"]:
                print(f"   Error: {result['stderr'][:200]}...")
    
    print(f"\n⏱️  Total time: {minutes}m {seconds}s")
    print(f"📊 Success rate: {success_count}/1 scripts")
    
    if success_count == 1:
        print("\n🎉 Rule cleaning completed successfully!")
        print("📋 Next steps:")
        print("   python 4_build_chromadb.py")
        return True
    else:
        print("\n❌ Rule cleaning failed. Check logs above.")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run parallel cleaning (step 3a)")
    
    # Change to script directory
    script_dir = Path(__file__).parent
    import os
    os.chdir(script_dir)
    
    # Run parallel cleaning
    success = asyncio.run(run_parallel_cleaning())
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
