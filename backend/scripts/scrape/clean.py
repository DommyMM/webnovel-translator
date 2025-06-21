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

def watch_and_clean(input_dir, output_dir, already_processed=None):
    """Watch the input directory and clean new files as they appear"""
    if already_processed is None:
        already_processed = set()
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    while True:
        # Get all chapter files
        chapter_files = list(input_dir.glob("chapter_*.txt"))
        
        # Process new files
        for file_path in chapter_files:
            if str(file_path) not in already_processed:
                print(f"Cleaning: {file_path.name}")
                if clean_chapter_file(file_path, output_dir):
                    already_processed.add(str(file_path))
                    print(f"✓ Cleaned: {file_path.name}")
        
        # Sleep for a short time
        time.sleep(2)

def main():
    """Main execution function"""
    print("🧹 Chinese Raw Chapter Cleaner")
    print("=" * 60)
    
    input_dir = "raw_chapters"
    output_dir = "clean_chapters"
    
    print(f"Watching directory: {input_dir}")
    print(f"Saving clean files to: {output_dir}")
    print("Press Ctrl+C to stop")
    
    # Start the watcher
    try:
        watch_and_clean(input_dir, output_dir)
    except KeyboardInterrupt:
        print("\nCleaning process stopped")

if __name__ == "__main__":
    main()