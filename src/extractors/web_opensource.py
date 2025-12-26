"""
Web scraping using open-source tools
"""
from bs4 import BeautifulSoup
import requests
from typing import Dict, List, Any
from urllib.parse import urljoin, urlparse


def scrape_with_beautifulsoup(url: str) -> Dict[str, Any]:
    """
    Scrape webpage content using BeautifulSoup
    
    Args:
        url: URL of the webpage to scrape
        
    Returns:
        Dictionary containing scraped content
    """
    result = {
        "tool": "BeautifulSoup",
        "url": url,
        "title": "",
        "text": "",
        "headings": [],
        "links": [],
        "images": [],
        "tables": [],
        "success": False,
        "error": None
    }
    
    try:
        # Fetch the webpage
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract title
        if soup.title:
            result["title"] = soup.title.string.strip()
        
        # Extract all text
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        result["text"] = soup.get_text(separator='\n', strip=True)
        
        # Extract headings
        for i in range(1, 7):
            headings = soup.find_all(f'h{i}')
            for heading in headings:
                result["headings"].append({
                    "level": i,
                    "text": heading.get_text(strip=True)
                })
        
        # Extract links
        for link in soup.find_all('a', href=True):
            absolute_url = urljoin(url, link['href'])
            result["links"].append({
                "text": link.get_text(strip=True),
                "url": absolute_url
            })
        
        # Extract images
        for img in soup.find_all('img'):
            img_url = img.get('src', '')
            if img_url:
                absolute_img_url = urljoin(url, img_url)
                result["images"].append({
                    "url": absolute_img_url,
                    "alt": img.get('alt', ''),
                    "title": img.get('title', '')
                })
        
        # Extract tables
        for table in soup.find_all('table'):
            rows = []
            for tr in table.find_all('tr'):
                cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                if cells:
                    rows.append(cells)
            if rows:
                result["tables"].append(rows)
        
        result["success"] = True
        
    except requests.exceptions.RequestException as e:
        result["error"] = f"Request error: {str(e)}"
    except Exception as e:
        result["error"] = f"Parsing error: {str(e)}"
    
    return result


if __name__ == "__main__":
    import json
    from datetime import datetime
    
    # Test with a simple webpage
    test_url = "https://en.wikipedia.org/wiki/Web_scraping"
    
    print("=" * 60)
    print("Testing BeautifulSoup...")
    print("=" * 60)
    result = scrape_with_beautifulsoup(test_url)
    print(f"Success: {result['success']}")
    
    if result['success']:
        print(f"Title: {result['title']}")
        print(f"Text length: {len(result['text'])} characters")
        print(f"Headings found: {len(result['headings'])}")
        print(f"Links found: {len(result['links'])}")
        print(f"Images found: {len(result['images'])}")
        print(f"Tables found: {len(result['tables'])}")
        print(f"\nFirst 3 headings:")
        for heading in result['headings'][:3]:
            print(f"  H{heading['level']}: {heading['text']}")
        
        # Save to text file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"data/scraped_content_{timestamp}.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("WEB SCRAPING RESULTS\n")
            f.write(f"URL: {result['url']}\n")
            f.write(f"Title: {result['title']}\n")
            f.write(f"Scraped at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # Full text content
            f.write("FULL TEXT CONTENT\n")
            f.write("-" * 80 + "\n")
            f.write(result['text'])
            f.write("\n\n")
            
            # Headings
            f.write("=" * 80 + "\n")
            f.write(f"HEADINGS ({len(result['headings'])} found)\n")
            f.write("=" * 80 + "\n")
            for heading in result['headings']:
                f.write(f"H{heading['level']}: {heading['text']}\n")
            f.write("\n")
            
            # Links
            f.write("=" * 80 + "\n")
            f.write(f"LINKS ({len(result['links'])} found)\n")
            f.write("=" * 80 + "\n")
            for i, link in enumerate(result['links'][:50], 1):
                f.write(f"{i}. {link['text']}\n   URL: {link['url']}\n")
            if len(result['links']) > 50:
                f.write(f"\n... and {len(result['links']) - 50} more links\n")
            f.write("\n")
            
            # Images
            f.write("=" * 80 + "\n")
            f.write(f"IMAGES ({len(result['images'])} found)\n")
            f.write("=" * 80 + "\n")
            for i, img in enumerate(result['images'], 1):
                f.write(f"{i}. URL: {img['url']}\n")
                if img['alt']:
                    f.write(f"   Alt: {img['alt']}\n")
                if img['title']:
                    f.write(f"   Title: {img['title']}\n")
            f.write("\n")
            
            # Tables
            if result['tables']:
                f.write("=" * 80 + "\n")
                f.write(f"TABLES ({len(result['tables'])} found)\n")
                f.write("=" * 80 + "\n")
                for i, table in enumerate(result['tables'], 1):
                    f.write(f"\nTable {i}:\n")
                    f.write("-" * 40 + "\n")
                    for row in table:
                        f.write(" | ".join(row) + "\n")
                    f.write("\n")
        
        print(f"\n✓ Content saved to: {output_file}")
        
        # Also save as JSON
        json_file = f"data/scraped_content_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"✓ JSON data saved to: {json_file}")
    else:
        print(f"Error: {result['error']}")