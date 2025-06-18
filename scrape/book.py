import re
from pathlib import Path
from bs4 import BeautifulSoup
import ebooklib
from ebooklib import epub

class EPUBChapterSplitter:
    def __init__(self, output_dir: str = "extracted_chapters"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def extract_text_from_epub(self, epub_path: str) -> str:
        """Extract all text content from EPUB file"""
        print(f"📖 Reading EPUB: {epub_path}")
        
        try:
            book = epub.read_epub(epub_path)
            full_text = ""
            
            # Get all HTML documents from the EPUB
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    content = item.get_content().decode('utf-8')
                    soup = BeautifulSoup(content, 'html.parser')
                    
                    # Extract text content
                    text_content = soup.get_text()
                    full_text += text_content + "\n"
            
            return full_text
            
        except Exception as e:
            print(f"❌ Error reading EPUB {epub_path}: {e}")
            return ""
    
    def split_chapters(self, full_text: str) -> dict:
        """Split text into individual chapters based on the pattern"""
        
        chapters = {}
        
        # Updated pattern to make translator part optional
        # This handles both formats:
        # 1. "Chapter X Title" appears twice, followed by "Translator: BornToBe"
        # 2. "Chapter X Title" appears twice (for later chapters)
        chapter_pattern = r'Chapter (\d+) ([^\n]+)\s*Chapter \1 \2(?:\s*Translator: BornToBe)?'
        
        # Find all chapter starts
        chapter_matches = list(re.finditer(chapter_pattern, full_text, re.MULTILINE))
        
        print(f"Found {len(chapter_matches)} chapter markers")
        
        for i, match in enumerate(chapter_matches):
            chapter_num = int(match.group(1))
            chapter_title = match.group(2).strip()
            
            # Find the start of content (after the chapter header)
            content_start = match.end()
            
            # Find the end of content (next chapter start or end of text)
            if i + 1 < len(chapter_matches):
                content_end = chapter_matches[i + 1].start()
            else:
                content_end = len(full_text)
            
            # Extract chapter content
            chapter_content = full_text[content_start:content_end].strip()
            
            # Clean up the content
            chapter_content = self.clean_chapter_content(chapter_content)
            
            if chapter_content:  # Only save non-empty chapters
                chapters[chapter_num] = {
                    'title': chapter_title,
                    'content': chapter_content,
                    'word_count': len(chapter_content.split())
                }
                
                print(f"✓ Chapter {chapter_num}: {chapter_title} ({len(chapter_content.split())} words)")
        
        return chapters
    
    def clean_chapter_content(self, content: str) -> str:
        """Clean up chapter content"""
        lines = content.split('\n')
        clean_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Skip lines that are clearly navigation or metadata
            skip_patterns = [
                r'^Chapter \d+',
                r'^Translator:',
                r'^\d+$',  # Just numbers
                r'^Previous Chapter',
                r'^Next Chapter',
                r'^Table of Contents',
                r'^Index',
            ]
            
            skip = False
            for pattern in skip_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    skip = True
                    break
            
            if not skip:
                clean_lines.append(line)
        
        return '\n\n'.join(clean_lines)
    
    def save_chapter(self, chapter_num: int, chapter_data: dict):
        """Save a single chapter to file"""
        filename = f"chapter_{chapter_num:04d}_en.txt"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Chapter {chapter_num}: {chapter_data['title']}\n")
            f.write(f"Word Count: {chapter_data['word_count']}\n")
            f.write("=" * 60 + "\n\n")
            f.write(chapter_data['content'])
    
    def process_epub_file(self, epub_path: str) -> dict:
        """Process a single EPUB file and return extracted chapters"""
        
        # Extract all text from EPUB
        full_text = self.extract_text_from_epub(epub_path)
        
        if not full_text:
            return {}
        
        # Split into chapters
        chapters = self.split_chapters(full_text)
        
        # Save each chapter
        for chapter_num, chapter_data in chapters.items():
            self.save_chapter(chapter_num, chapter_data)
        
        return chapters
    
    def process_all_epubs(self, epub_directory: str):
        """Process all EPUB files in a directory"""
        epub_dir = Path(epub_directory)
        
        if not epub_dir.exists():
            print(f"❌ Directory not found: {epub_directory}")
            return
        
        # Find all EPUB files
        epub_files = list(epub_dir.glob("*.epub"))
        epub_files.sort()  # Process in order
        
        if not epub_files:
            print(f"❌ No EPUB files found in {epub_directory}")
            return
        
        print(f"📚 Found {len(epub_files)} EPUB files to process")
        
        all_chapters = {}
        
        for epub_file in epub_files:
            print(f"\n🔄 Processing: {epub_file.name}")
            chapters = self.process_epub_file(str(epub_file))
            all_chapters.update(chapters)
        
        # Summary
        print(f"\n🎉 Processing Complete!")
        print(f"📊 Total chapters extracted: {len(all_chapters)}")
        
        if all_chapters:
            min_chapter = min(all_chapters.keys())
            max_chapter = max(all_chapters.keys())
            print(f"📖 Chapter range: {min_chapter} - {max_chapter}")
            
            # Check for missing chapters in sequence
            missing_chapters = []
            for i in range(min_chapter, max_chapter + 1):
                if i not in all_chapters:
                    missing_chapters.append(i)
            
            if missing_chapters:
                print(f"⚠️ Missing chapters: {len(missing_chapters)}")
                # Log the first 10 missing chapters as an example
                if len(missing_chapters) > 10:
                    print(f"   First 10 missing: {missing_chapters[:10]}...")
                else:
                    print(f"   Missing: {missing_chapters}")
        
        # Save summary
        summary_file = self.output_dir / "extraction_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("EPUB Extraction Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total chapters: {len(all_chapters)}\n")
            f.write(f"Chapter range: {min_chapter} - {max_chapter}\n\n")
            
            f.write("Chapter List:\n")
            for chapter_num in sorted(all_chapters.keys()):
                chapter_data = all_chapters[chapter_num]
                f.write(f"Chapter {chapter_num:4d}: {chapter_data['title']} ({chapter_data['word_count']} words)\n")
        
        print(f"📄 Summary saved to: {summary_file}")
        
        return all_chapters

def main():
    """Main execution function"""
    
    # Your EPUB directory path
    epub_directory = r"C:\Users\domin\Downloads\Webnovel\tl"
    
    # Output directory for extracted chapters
    output_directory = "nine_star_chapters"
    
    print("🚀 Nine Star Hegemon Body Arts - EPUB Chapter Splitter")
    print("=" * 60)
    
    # Initialize splitter
    splitter = EPUBChapterSplitter(output_directory)
    
    # Process all EPUB files
    chapters = splitter.process_all_epubs(epub_directory)
    
    print(f"\n✅ All chapters saved to: {Path(output_directory).absolute()}")
    
    # Quick validation
    if chapters:
        print(f"\n🔍 Quick validation:")
        sample_chapters = list(sorted(chapters.keys()))[:5]  # First 5 chapters
        
        for chapter_num in sample_chapters:
            chapter_data = chapters[chapter_num]
            print(f"  Chapter {chapter_num}: {chapter_data['title'][:50]}...")
            print(f"    Words: {chapter_data['word_count']}")
            print(f"    Preview: {chapter_data['content'][:100]}...")
            print()

if __name__ == "__main__":
    main()