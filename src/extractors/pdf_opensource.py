"""
PDF extraction using open-source tools
"""
import PyPDF2
import pdfplumber
from pathlib import Path
from typing import Dict, List, Any


def extract_with_pypdf2(pdf_path: str) -> Dict[str, Any]:
    """
    Extract text and metadata from PDF using PyPDF2
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary containing extracted text and metadata
    """
    result = {
        "tool": "PyPDF2",
        "text": "",
        "metadata": {},
        "pages": [],
        "success": False,
        "error": None
    }
    
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Extract metadata
            metadata = pdf_reader.metadata
            if metadata:
                result["metadata"] = {
                    "title": metadata.get('/Title', 'N/A'),
                    "author": metadata.get('/Author', 'N/A'),
                    "pages": len(pdf_reader.pages)
                }
            
            # Extract text from each page
            full_text = []
            for page_num, page in enumerate(pdf_reader.pages, 1):
                text = page.extract_text()
                result["pages"].append({
                    "page_number": page_num,
                    "text": text,
                    "char_count": len(text)
                })
                full_text.append(text)
            
            result["text"] = "\n\n".join(full_text)
            result["success"] = True
            
    except Exception as e:
        result["error"] = str(e)
    
    return result


def extract_with_pdfplumber(pdf_path: str) -> Dict[str, Any]:
    """
    Extract text, tables, and images from PDF using pdfplumber
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary containing extracted content
    """
    result = {
        "tool": "pdfplumber",
        "text": "",
        "tables": [],
        "images": [],
        "pages": [],
        "success": False,
        "error": None
    }
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            full_text = []
            
            for page_num, page in enumerate(pdf.pages, 1):
                page_data = {
                    "page_number": page_num,
                    "text": "",
                    "tables": [],
                    "images": []
                }
                
                # Extract text
                text = page.extract_text()
                if text:
                    page_data["text"] = text
                    full_text.append(text)
                
                # Extract tables
                tables = page.extract_tables()
                if tables:
                    page_data["tables"] = tables
                    result["tables"].extend([{
                        "page": page_num,
                        "data": table
                    } for table in tables])
                
                # Extract images info
                images = page.images
                if images:
                    page_data["images"] = [{
                        "x0": img["x0"],
                        "y0": img["y0"],
                        "x1": img["x1"],
                        "y1": img["y1"],
                        "width": img["width"],
                        "height": img["height"]
                    } for img in images]
                    result["images"].extend(page_data["images"])
                
                result["pages"].append(page_data)
            
            result["text"] = "\n\n".join(full_text)
            result["success"] = True
            
    except Exception as e:
        result["error"] = str(e)
    
    return result


if __name__ == "__main__":
    # Test the functions
    sample_pdf = "data/sample_pdfs/sample.pdf"
    
    print("=" * 60)
    print("Testing PyPDF2...")
    print("=" * 60)
    result1 = extract_with_pypdf2(sample_pdf)
    print(f"Success: {result1['success']}")
    if result1['success']:
        print(f"Text length: {len(result1['text'])} characters")
        print(f"Pages: {len(result1['pages'])}")
        print(f"First 200 characters: {result1['text'][:200]}")
        print(f"\n--- FULL TEXT ---")
        print(result1['text'])
    else:
        print(f"Error: {result1['error']}")
    
    print("\n" + "=" * 60)
    print("Testing pdfplumber...")
    print("=" * 60)
    result2 = extract_with_pdfplumber(sample_pdf)
    print(f"Success: {result2['success']}")
    if result2['success']:
        print(f"Text length: {len(result2['text'])} characters")
        print(f"Tables found: {len(result2['tables'])}")
        print(f"Images found: {len(result2['images'])}")
        print(f"First 200 characters: {result2['text'][:200]}")
        print(f"\n--- FULL TEXT ---")
        print(result1['text'])
    else:
        print(f"Error: {result2['error']}")