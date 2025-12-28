"""
FastAPI Backend for Document Processing
"""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional
import sys
from pathlib import Path
import shutil
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.extractors.pdf_opensource import extract_with_pypdf2, extract_with_pdfplumber
from src.extractors.pdf_enterprise import extract_with_azure
from src.extractors.web_opensource import scrape_with_beautifulsoup
from src.converters.docling_converter import convert_with_docling
from src.converters.markitdown_converter import convert_with_markitdown
from src.storage.s3_manager import S3Manager

app = FastAPI(title="AI Document Processor API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize S3 Manager
s3 = S3Manager()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Document Processor API",
        "version": "1.0.0",
        "endpoints": {
            "pdf": "/process-pdf",
            "web": "/process-webpage",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/process-pdf")
async def process_pdf(
    file: UploadFile = File(...),
    method: str = Form("opensource"),
    converter: str = Form("markitdown"),
    upload_to_s3: bool = Form(True)
):
    """
    Process PDF file
    
    Args:
        file: PDF file to process
        method: 'opensource' (PyPDF2/pdfplumber) or 'enterprise' (Azure)
        converter: 'markitdown' or 'docling'
        upload_to_s3: Whether to upload results to S3
    """
    try:
        # Save uploaded file temporarily
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_dir = Path("data/temp")
        temp_dir.mkdir(exist_ok=True, parents=True)
        
        temp_pdf_path = temp_dir / f"{timestamp}_{file.filename}"
        with open(temp_pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Check file size
        file_size_mb = temp_pdf_path.stat().st_size / (1024 * 1024)
        
        # Azure has 4MB limit for free tier, auto-fallback to opensource for large files
        if method == "enterprise" and file_size_mb > 4:
            print(f"File too large for Azure ({file_size_mb:.2f}MB > 4MB), falling back to opensource")
            method = "opensource"
            fallback_message = f"File size ({file_size_mb:.2f}MB) exceeds Azure limit (4MB). Using open-source extraction instead."
        else:
            fallback_message = None
        
        # Extract content based on method
        if method == "enterprise":
            extraction_result = extract_with_azure(str(temp_pdf_path))
        else:
            # Use pdfplumber for better extraction
            extraction_result = extract_with_pdfplumber(str(temp_pdf_path))
        
        if not extraction_result["success"]:
            raise HTTPException(status_code=500, detail=extraction_result["error"])
        
        # Convert to markdown
        if converter == "docling":
            conversion_result = convert_with_docling(str(temp_pdf_path))
        else:
            conversion_result = convert_with_markitdown(str(temp_pdf_path))
        
        # Save markdown
        # Save markdown
        md_filename = f"{temp_pdf_path.stem}_converted.md"
        md_path = temp_dir / md_filename
        markdown_content = conversion_result.get("markdown", extraction_result.get("text", ""))

        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        # Upload to S3 if requested
        s3_result = None
        if upload_to_s3:
            s3_key = s3.generate_s3_key("pdfs", "processed", md_filename)
            s3_result = s3.upload_file(
                str(md_path),
                s3_key,
                metadata={
                    'method': method,
                    'converter': converter,
                    'file_size_mb': str(file_size_mb),
                    'processed_date': datetime.now().isoformat()
                }
            )
        
        # Clean up temp files
        temp_pdf_path.unlink()
        
        response_data = {
            "success": True,
            "extraction": {
                "method": method,
                "pages": len(extraction_result.get("pages", [])),
                "tables": len(extraction_result.get("tables", [])),
                "text_length": len(extraction_result.get("text", ""))
            },
            "conversion": {
                "converter": converter,
                "markdown_length": len(conversion_result.get("markdown", ""))
            },
            "s3_upload": s3_result if upload_to_s3 else None,
            "markdown_file": str(md_path),
            "markdown_content": markdown_content,  # Add this line
            "file_size_mb": round(file_size_mb, 2)
        }
        
        if fallback_message:
            response_data["warning"] = fallback_message
        
        return JSONResponse(response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process-webpage")
async def process_webpage(
    url: str = Form(...),
    converter: str = Form("markitdown"),
    upload_to_s3: bool = Form(True)
):
    """
    Process webpage URL
    
    Args:
        url: URL of webpage to scrape
        converter: 'markitdown' or 'docling'
        upload_to_s3: Whether to upload results to S3
    """
    try:
        # Scrape webpage
        scrape_result = scrape_with_beautifulsoup(url)
        
        if not scrape_result["success"]:
            raise HTTPException(status_code=500, detail=scrape_result["error"])
        
        # Save scraped content as JSON temporarily
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_dir = Path("data/temp")
        temp_dir.mkdir(exist_ok=True, parents=True)
        
        import json
        json_path = temp_dir / f"scraped_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(scrape_result, f, indent=2)
        
        # Create markdown from scraped content
        markdown_parts = []
        markdown_parts.append(f"# {scrape_result.get('title', 'Web Page')}\n")
        markdown_parts.append(f"**Source:** {url}\n\n")
        markdown_parts.append("## Content\n")
        markdown_parts.append(scrape_result.get('text', ''))
        
        markdown_content = "\n".join(markdown_parts)
        
        # Save markdown
        md_filename = f"webpage_{timestamp}.md"
        md_path = temp_dir / md_filename
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        # Upload to S3 if requested
        s3_result = None
        if upload_to_s3:
            s3_key = s3.generate_s3_key("webpages", "processed", md_filename)
            s3_result = s3.upload_file(
                str(md_path),
                s3_key,
                metadata={
                    'source_url': url,
                    'converter': converter,
                    'processed_date': datetime.now().isoformat()
                }
            )
        
        return JSONResponse({
            "success": True,
            "scraping": {
                "url": url,
                "title": scrape_result.get("title", ""),
                "text_length": len(scrape_result.get("text", "")),
                "headings": len(scrape_result.get("headings", [])),
                "links": len(scrape_result.get("links", [])),
                "images": len(scrape_result.get("images", [])),
                "tables": len(scrape_result.get("tables", []))
            },
            "s3_upload": s3_result if upload_to_s3 else None,
            "markdown_file": str(md_path)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)