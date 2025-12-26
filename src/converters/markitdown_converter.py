"""
Convert extracted content to Markdown using MarkItDown
"""
import json
from pathlib import Path
from typing import Dict, Any
from markitdown import MarkItDown


def convert_with_markitdown(input_path: str, output_path: str = None) -> Dict[str, Any]:
    """
    Convert file to Markdown using MarkItDown
    
    Args:
        input_path: Path to input file (PDF, JSON, etc.)
        output_path: Optional path for output markdown
        
    Returns:
        Dictionary with conversion results
    """
    result = {
        "tool": "MarkItDown",
        "input_file": input_path,
        "output_file": output_path,
        "markdown": "",
        "success": False,
        "error": None
    }
    
    try:
        md = MarkItDown()
        
        # Convert file
        markdown_result = md.convert(input_path)
        result["markdown"] = markdown_result.text_content
        
        # Save if output path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result["markdown"])
            result["output_file"] = output_path
        
        result["success"] = True
        
    except Exception as e:
        result["error"] = str(e)
    
    return result


if __name__ == "__main__":
    from datetime import datetime
    
    # Test with PDF
    pdf_path = "data/sample_pdfs/sample.pdf"
    
    if Path(pdf_path).exists():
        print("=" * 60)
        print("Converting PDF with MarkItDown...")
        print("=" * 60)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_md = f"data/converted_markitdown_{timestamp}.md"
        
        result = convert_with_markitdown(pdf_path, output_md)
        
        if result['success']:
            print(f"✓ Success!")
            print(f"✓ Output: {output_md}")
            print(f"✓ Length: {len(result['markdown'])} characters")
            print(f"\nMarkdown content:")
            print("-" * 60)
            print(result['markdown'])
        else:
            print(f"✗ Error: {result['error']}")
    else:
        print(f"PDF not found: {pdf_path}")