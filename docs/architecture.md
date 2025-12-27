# System Architecture

## High-Level Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                         USER LAYER                          │
│                                                               │
│  ┌──────────────────┐              ┌──────────────────┐    │
│  │   Web Browser    │              │   API Clients    │    │
│  │  (Streamlit UI)  │              │  (curl, Postman) │    │
│  └────────┬─────────┘              └────────┬─────────┘    │
│           │                                  │               │
└───────────┼──────────────────────────────────┼──────────────┘
            │                                  │
            └──────────────┬───────────────────┘
                           │ HTTP/REST
┌──────────────────────────▼──────────────────────────────────┐
│                     APPLICATION LAYER                        │
│                                                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              FastAPI Backend (main.py)                │  │
│  │  - /process-pdf endpoint                              │  │
│  │  - /process-webpage endpoint                          │  │
│  │  - /health endpoint                                   │  │
│  └───────────┬───────────────────────────────────────────┘  │
│              │                                               │
└──────────────┼───────────────────────────────────────────────┘
               │
    ┌──────────┴──────────┐
    │                     │
    ▼                     ▼
┌──────────┐         ┌──────────┐
│   PDF    │         │   Web    │
│Extractor │         │ Scraper  │
└────┬─────┘         └────┬─────┘
     │                    │
     └──────────┬─────────┘
                │
┌───────────────▼───────────────────────────────────────────┐
│                    PROCESSING LAYER                        │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   PyPDF2     │  │  pdfplumber  │  │  Beautiful   │   │
│  │              │  │              │  │    Soup      │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
│                                                             │
│  ┌──────────────────────────────────────────────────┐    │
│  │     Azure Document Intelligence (Enterprise)      │    │
│  └──────────────────────────────────────────────────┘    │
│                                                             │
│  ┌──────────────┐              ┌──────────────┐          │
│  │  MarkItDown  │              │   Docling    │          │
│  │  Converter   │              │  Converter   │          │
│  └──────────────┘              └──────────────┘          │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────┐
│                     STORAGE LAYER                         │
│                                                            │
│  ┌────────────────────────────────────────────────────┐  │
│  │              AWS S3 Bucket                         │  │
│  │                                                      │  │
│  │  /pdfs/processed/2025-12/document.md               │  │
│  │  /webpages/processed/2025-12/webpage.md            │  │
│  │  /images/processed/2025-12/image.jpg               │  │
│  │  /markdown/processed/2025-12/content.md            │  │
│  │                                                      │  │
│  │  Metadata: source, tool, date, type                │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

## Data Flow

### PDF Processing Flow
```
1. User uploads PDF via Streamlit
2. Streamlit sends file to FastAPI
3. FastAPI saves file temporarily
4. Extract content using selected method:
   - opensource: PyPDF2 or pdfplumber
   - enterprise: Azure Document Intelligence
5. Convert to Markdown using:
   - MarkItDown (fast)
   - Docling (advanced)
6. Upload to S3 with metadata
7. Return results to user
8. Display in Streamlit UI
```

### Web Scraping Flow
```
1. User enters URL in Streamlit
2. Streamlit sends URL to FastAPI
3. FastAPI scrapes with BeautifulSoup
4. Extract: text, headings, links, images, tables
5. Save as structured JSON
6. Convert to Markdown
7. Upload to S3 with metadata
8. Return results to user
9. Display in Streamlit UI
```

## Component Details

### Extractors
- **pdf_opensource.py**: PyPDF2 and pdfplumber implementations
- **pdf_enterprise.py**: Azure Document Intelligence integration
- **web_opensource.py**: BeautifulSoup web scraper

### Converters
- **markitdown_converter.py**: Fast, lightweight markdown conversion
- **docling_converter.py**: Advanced document understanding

### Storage
- **s3_manager.py**: Handles S3 uploads, organized structure, metadata

### API
- **main.py**: FastAPI application with REST endpoints

### Frontend
- **app.py**: Streamlit user interface

## Technology Stack

| Layer | Technologies |
|-------|-------------|
| Frontend | Streamlit |
| Backend | FastAPI, Uvicorn |
| PDF Processing | PyPDF2, pdfplumber, Azure DI |
| Web Scraping | BeautifulSoup, requests |
| Markdown | Docling, MarkItDown |
| Storage | AWS S3, boto3 |
| Language | Python 3.13 |

## Security & Configuration

- Environment variables stored in `.env`
- AWS credentials secured
- Azure API keys protected
- CORS enabled for frontend-backend communication
- S3 bucket access controls