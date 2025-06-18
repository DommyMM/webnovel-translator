import requests
from bs4 import BeautifulSoup
import time
import os
import re
from pathlib import Path

def scrape_chapters(start_url, output_dir, max_chapters=100, start_chapter=1):
    """Scrape chapters from the Chinese novel website."""
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    current_url = start_url
    chapter_count = start_chapter - 1  # Adjust the starting count
    start_time = time.time()
    
    while current_url and chapter_count < max_chapters:
        # Fetch the page
        print(f"Fetching: {current_url}")
        try:
            # Add more robust error handling
            response = requests.get(current_url, timeout=15)  # Increased timeout
            response.encoding = 'utf-8'  # Ensure correct encoding
            
            if response.status_code != 200:
                print(f"Failed to fetch {current_url}, status code: {response.status_code}")
                break
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract chapter title
            title_elem = soup.find('h1')
            title = title_elem.text.strip() if title_elem else f"Chapter {chapter_count+1}"
            
            # Use the correct selector with simplified extraction
            content_elem = soup.find('div', id='content', class_='panel-body')
            
            if content_elem:
                # Simplify content extraction - get all text
                content = content_elem.get_text('\n')
                
                # Better cleaning
                # Remove ads and other unwanted content
                content = re.sub(r'adstart.*?adend', '', content, flags=re.DOTALL)
                content = re.sub(r'斗破小说网.*?阅读', '', content)
                content = re.sub(r'最新网址：www\..*?\.org', '', content)
                content = re.sub(r'手机阅读.*?org', '', content)
                
                # Save the chapter
                filename = f"chapter_{chapter_count+1:04d}_cn.txt"
                filepath = output_dir / filename
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(f"{title}\n\n{content}")
                
                chapter_count += 1
                
                # Show progress with elapsed time and speed
                elapsed = time.time() - start_time
                speed = chapter_count / elapsed if elapsed > 0 else 0
                print(f"✓ Saved: {filename} [{chapter_count} chapters, {speed:.2f} ch/sec]")
            else:
                print("❌ Content not found on page")
            
            # Find the next chapter link
            next_link = None
            for li in soup.find_all('li', class_='col-md-4 col-xs-12 col-sm-12'):
                if '下一章' in li.text:
                    next_a = li.find('a')
                    if next_a and next_a.get('href'):
                        next_link = 'https://www.doupocangqiong.org' + next_a['href']
                        break
            
            if not next_link:
                print("⚠️ No next chapter link found. Stopping.")
                break
            
            current_url = next_link
            
            # Slightly reduced wait time but still being respectful
            time.sleep(0.5)
            
        except Exception as e:
            print(f"❌ Error processing {current_url}: {e}")
            # Try again after a short pause
            time.sleep(2)
            continue
    
    # Final stats
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    
    print(f"\n🎉 Scraping completed in {minutes}m {seconds}s")
    print(f"📊 Total chapters scraped: {chapter_count - start_chapter + 1}")
    if chapter_count > 0:
        print(f"⚡ Average speed: {(chapter_count - start_chapter + 1) / elapsed:.2f} chapters per second")
    
    return chapter_count - start_chapter + 1

def main():
    """Main execution function"""
    print("🚀 Nine Star Hegemon Body Arts - Chinese Raw Scraper")
    print("=" * 60)
    
    # UPDATED: Continue from where we left off
    start_url = "https://www.doupocangqiong.org/jiuxingbatijue/12117316.html"
    output_dir = "raw_chapters"
    max_chapters = 5700
    
    # Update the starting chapter number
    start_chapter = 2367  # Since you've completed up to 2366
    
    print(f"📚 Resuming scrape from: {start_url} (Chapter {start_chapter})")
    print(f"📁 Saving chapters to: {Path(output_dir).absolute()}\n")
    
    # Modified call to account for starting chapter number
    total_chapters = scrape_chapters(start_url, output_dir, max_chapters, start_chapter)
    
    print(f"\n✅ All raw chapters saved to: {Path(output_dir).absolute()}")

if __name__ == "__main__":
    main()