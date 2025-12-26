"""
PDF extraction using Azure Document Intelligence (Enterprise)
"""
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()


def extract_with_azure(pdf_path: str) -> Dict[str, Any]:
    """
    Extract and convert PDF to Markdown using Azure Document Intelligence
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary containing extracted content and markdown
    """
    result = {
        "tool": "Azure Document Intelligence",
        "text": "",
        "markdown": "",
        "tables": [],
        "pages": [],
        "success": False,
        "error": None
    }
    
    try:
        # Initialize Azure client
        endpoint = os.getenv("AZURE_ENDPOINT")
        key = os.getenv("AZURE_KEY")
        
        if not endpoint or not key:
            result["error"] = "Azure credentials not found in .env file"
            return result
        
        client = DocumentAnalysisClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key)
        )
        
        # Read PDF file
        with open(pdf_path, "rb") as f:
            poller = client.begin_analyze_document(
                "prebuilt-layout", document=f
            )
        
        # Get results
        azure_result = poller.result()
        
        # Extract text from pages
        full_text = []
        markdown_parts = []
        
        for page in azure_result.pages:
            page_data = {
                "page_number": page.page_number,
                "width": page.width,
                "height": page.height,
                "text": ""
            }
            
            # Extract text
            page_text = []
            for line in page.lines:
                page_text.append(line.content)
            
            page_data["text"] = "\n".join(page_text)
            full_text.append(page_data["text"])
            result["pages"].append(page_data)
        
        result["text"] = "\n\n".join(full_text)
        
        # Extract tables
        for table in azure_result.tables:
            table_data = []
            for cell in table.cells:
                if cell.row_index >= len(table_data):
                    table_data.append([])
                while cell.column_index >= len(table_data[cell.row_index]):
                    table_data[cell.row_index].append("")
                table_data[cell.row_index][cell.column_index] = cell.content
            
            result["tables"].append({
                "row_count": table.row_count,
                "column_count": table.column_count,
                "data": table_data
            })
        
        # Generate Markdown
        markdown_parts.append(f"# Document Analysis\n")
        markdown_parts.append(f"**Pages:** {len(result['pages'])}\n")
        markdown_parts.append(f"**Tables:** {len(result['tables'])}\n\n")
        
        # Add content
        markdown_parts.append("## Content\n")
        markdown_parts.append(result["text"])
        markdown_parts.append("\n\n")
        
        # Add tables in markdown format
        if result["tables"]:
            markdown_parts.append("## Tables\n\n")
            for i, table in enumerate(result["tables"], 1):
                markdown_parts.append(f"### Table {i}\n\n")
                table_data = table["data"]
                if table_data:
                    # Header
                    markdown_parts.append("| " + " | ".join(table_data[0]) + " |")
                    markdown_parts.append("|" + "|".join(["---"] * len(table_data[0])) + "|")
                    # Rows
                    for row in table_data[1:]:
                        markdown_parts.append("| " + " | ".join(row) + " |")
                    markdown_parts.append("\n")
        
        result["markdown"] = "\n".join(markdown_parts)
        result["success"] = True
        
    except Exception as e:
        result["error"] = str(e)
    
    return result


if __name__ == "__main__":
    from datetime import datetime
    
    # Test with sample PDF
    sample_pdf = "data/sample_pdfs/sample.pdf"
    
    print("=" * 60)
    print("Testing Azure Document Intelligence...")
    print("=" * 60)
    
    result = extract_with_azure(sample_pdf)
    
    if result['success']:
        print(f"✓ Success!")
        print(f"Text length: {len(result['text'])} characters")
        print(f"Pages: {len(result['pages'])}")
        print(f"Tables: {len(result['tables'])}")
        print(f"\nMarkdown Preview:")
        print("-" * 60)
        print(result['markdown'][:500])
        
        # Save markdown
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"data/azure_extracted_{timestamp}.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result['markdown'])
        print(f"\n✓ Markdown saved to: {output_file}")
    else:
        print(f"✗ Error: {result['error']}")