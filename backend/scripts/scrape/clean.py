import os
import re
import time
from pathlib import Path
import shutil

def clean_chapter_file(file_path, output_dir):
    """Clean a single chapter file by removing navigation and site text"""
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Remove everything between "上一章：" and "斗破小说网"
        content = re.sub(r'上一章：.*?斗破小说网,\s*www\.doupocangqiong\.org\s*,.*?阅读\.', 
                         '', content, flags=re.DOTALL)
        
        # Remove any remaining site information
        content = re.sub(r'斗破小说网.*?阅读', '', content)
        content = re.sub(r'最新网址：www\..*?\.org', '', content)
        content = re.sub(r'手机阅读.*?org', '', content)
        
        # Remove any extra blank lines (more than 2 consecutive)
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # Get the original filename
        filename = os.path.basename(file_path)
        output_path = os.path.join(output_dir, filename)
        
        # Write the cleaned content
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content.strip())
        
        return True
    
    except Exception as e:
        print(f"Error cleaning {file_path}: {e}")
        return False

def get_already_cleaned_files(output_dir):
    """Get list of files that have already been cleaned"""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return set()
    
    cleaned_files = set()
    for file_path in output_dir.glob("chapter_*.txt"):
        cleaned_files.add(file_path.name)
    
    return cleaned_files

def watch_and_clean(input_dir, output_dir, idle_timeout=5, already_processed=None):
    """Watch the input directory and clean new files as they appear"""
    if already_processed is None:
        already_processed = set()
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Get list of already cleaned files
    already_cleaned = get_already_cleaned_files(output_dir)
    print(f"Found {len(already_cleaned)} already cleaned files")
    
    last_activity_time = time.time()
    files_processed_this_session = 0
    
    while True:
        # Get all chapter files
        chapter_files = list(input_dir.glob("chapter_*.txt"))
        files_processed_this_cycle = 0
        
        # Process new files
        for file_path in chapter_files:
            filename = file_path.name
            file_path_str = str(file_path)
            
            # Skip if already cleaned or already processed this session
            if filename in already_cleaned or file_path_str in already_processed:
                continue
                
            print(f"Cleaning: {filename}")
            if clean_chapter_file(file_path, output_dir):
                already_processed.add(file_path_str)
                already_cleaned.add(filename)
                files_processed_this_cycle += 1
                files_processed_this_session += 1
                last_activity_time = time.time()
                print(f"✓ Cleaned: {filename}")
        
        # Check if we processed any files this cycle
        if files_processed_this_cycle > 0:
            print(f"Processed {files_processed_this_cycle} files this cycle. Total this session: {files_processed_this_session}")
        
        # Check for idle timeout
        time_since_last_activity = time.time() - last_activity_time
        if time_since_last_activity >= idle_timeout:
            print(f"\n✅ No new files to process for {idle_timeout} seconds.")
            print(f"📊 Total files cleaned this session: {files_processed_this_session}")
            print("🎉 Cleaning process completed!")
            break
        
        # Sleep for a short time
        time.sleep(2)

def main():
    """Main execution function"""
    print("🧹 Chinese Raw Chapter Cleaner")
    print("=" * 60)
    
    input_dir = r"c:\Users\domin\Downloads\Webnovel\backend\data\chapters\raw"
    output_dir = r"c:\Users\domin\Downloads\Webnovel\backend\data\chapters\clean"
    
    print(f"Watching directory: {input_dir}")
    print(f"Saving clean files to: {output_dir}")
    print("Will auto-exit after 5 seconds of no activity")
    print("Press Ctrl+C to stop manually")
    
    # Start the watcher
    try:
        watch_and_clean(input_dir, output_dir, idle_timeout=5)
    except KeyboardInterrupt:
        print("\nCleaning process stopped manually")

if __name__ == "__main__":
    main()