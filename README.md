# 🤖 AI Document Processor

An intelligent document processing system that extracts, converts, and stores content from PDFs and webpages using both open-source and enterprise AI tools.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Tool Comparison](#tool-comparison)
- [Project Structure](#project-structure)
- [Team Contributions](#team-contributions)
- [Deployment](#deployment)
- [License](#license)

## 🎯 Overview

This project evaluates and compares multiple document processing approaches to help determine the best solution for extracting data from unstructured sources. It provides a complete pipeline from extraction to storage with both open-source and enterprise solutions.

**Key Objectives:**
- Compare open-source vs enterprise document processing tools
- Evaluate Docling vs MarkItDown for markdown conversion
- Implement structured S3 storage with metadata
- Provide user-friendly interfaces for document processing

## ✨ Features

### Document Processing
- ✅ **PDF Extraction**
  - Open-source: PyPDF2, pdfplumber
  - Enterprise: Azure Document Intelligence
  - Extract text, tables, images, and metadata

- ✅ **Web Scraping**
  - BeautifulSoup for HTML parsing
  - Extract headings, links, images, tables
  - Save structured JSON data

- ✅ **Markdown Conversion**
  - Docling: Advanced document understanding
  - MarkItDown: Fast, lightweight conversion
  - Preserve document structure and formatting

### Storage & Organization
- ✅ **AWS S3 Integration**
  - Organized folder structure by type and date
  - Metadata tagging for searchability
  - Automatic upload after processing

### APIs & Interfaces
- ✅ **FastAPI Backend**
  - RESTful API endpoints
  - File upload handling
  - Async processing

- ✅ **Streamlit Frontend**
  - Drag-and-drop file upload
  - Real-time processing feedback
  - View and download results
  - S3 file browser

## 🏗️ Architecture
```
┌─────────────┐
│  Streamlit  │ ← User Interface
│   Frontend  │
└──────┬──────┘
       │ HTTP
       ▼
┌─────────────┐
│   FastAPI   │ ← REST API
│   Backend   │
└──────┬──────┘
       │
   ┌───┴────────────────┐
   │                    │
   ▼                    ▼
┌──────────┐      ┌──────────┐
│  PDF     │      │   Web    │
│Extractors│      │ Scrapers │
└────┬─────┘      └────┬─────┘
     │                 │
     └────────┬────────┘
              ▼
       ┌─────────────┐
       │  Markdown   │
       │ Converters  │
       └──────┬──────┘
              ▼
       ┌─────────────┐
       │   AWS S3    │
       │   Storage   │
       └─────────────┘
```

## 🛠️ Tech Stack

### Core Technologies
- **Python 3.13**
- **FastAPI** - Modern web framework for APIs
- **Streamlit** - Interactive web applications
- **AWS S3** - Cloud object storage
- **Azure Document Intelligence** - Enterprise AI extraction

### Libraries
- **PDF Processing:** PyPDF2, pdfplumber
- **Web Scraping:** BeautifulSoup4, requests
- **Markdown Conversion:** Docling, MarkItDown
- **Cloud Storage:** boto3 (AWS SDK)
- **API Development:** uvicorn, python-multipart

## 📥 Installation

### Prerequisites
- Python 3.11+
- AWS Account (for S3)
- Azure Account (optional, for Document Intelligence)
- Git

### Setup Instructions

1. **Clone the repository**
```bash
git clone git@github.com:DarshanGaviyappa/ai-document-processor.git
cd ai-document-processor
```

2. **Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
```bash
# Create .env file
cp .env.example .env

# Edit .env with your credentials
# - AWS_ACCESS_KEY_ID
# - AWS_SECRET_ACCESS_KEY
# - S3_BUCKET_NAME
# - AZURE_ENDPOINT (optional)
# - AZURE_KEY (optional)
```

5. **Create S3 Bucket**
```bash
aws s3 mb s3://your-bucket-name
```

## 🚀 Usage

### Running the Application

**Terminal 1 - Start FastAPI Backend:**
```bash
source venv/bin/activate
python src/api/main.py
```

**Terminal 2 - Start Streamlit Frontend:**
```bash
source venv/bin/activate
streamlit run streamlit_app/app.py
```

### Access Points
- **Streamlit UI:** http://localhost:8501
- **FastAPI Docs:** http://localhost:8000/docs
- **API Health:** http://localhost:8000/health

### Processing Documents

#### Via Streamlit UI:
1. Navigate to http://localhost:8501
2. Choose "PDF Upload" or "Web Scraping" tab
3. Select extraction method and converter
4. Upload file or enter URL
5. Click "Process" and view results

#### Via API:
```bash
# Process PDF
curl -X POST "http://localhost:8000/process-pdf" \
  -F "file=@document.pdf" \
  -F "method=opensource" \
  -F "converter=markitdown" \
  -F "upload_to_s3=true"

# Process Webpage
curl -X POST "http://localhost:8000/process-webpage" \
  -F "url=https://example.com" \
  -F "converter=docling" \
  -F "upload_to_s3=true"
```

## 📚 API Documentation

### Endpoints

#### `GET /`
Root endpoint with API information

#### `GET /health`
Health check endpoint

#### `POST /process-pdf`
Process PDF file

**Parameters:**
- `file` (file): PDF file to process
- `method` (string): 'opensource' or 'enterprise'
- `converter` (string): 'markitdown' or 'docling'
- `upload_to_s3` (boolean): Upload results to S3

**Response:**
```json
{
  "success": true,
  "extraction": {
    "method": "opensource",
    "pages": 5,
    "tables": 2,
    "text_length": 1500
  },
  "conversion": {
    "converter": "markitdown",
    "markdown_length": 1800
  },
  "s3_upload": {
    "success": true,
    "s3_key": "pdfs/processed/2025-12/document.md",
    "s3_url": "s3://bucket-name/pdfs/processed/2025-12/document.md"
  }
}
```

#### `POST /process-webpage`
Scrape and process webpage

**Parameters:**
- `url` (string): URL to scrape
- `converter` (string): 'markitdown' or 'docling'
- `upload_to_s3` (boolean): Upload results to S3

## 📊 Tool Comparison

See [docs/tool_comparison.md](docs/tool_comparison.md) for detailed comparison.

### Quick Summary

| Tool | Cost | Accuracy | Speed | Use Case |
|------|------|----------|-------|----------|
| PyPDF2 | Free | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Simple PDFs |
| pdfplumber | Free | ⭐⭐⭐⭐ | ⭐⭐⭐ | PDFs with tables |
| Azure DI | Paid | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Complex/scanned PDFs |
| BeautifulSoup | Free | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Web scraping |
| MarkItDown | Free | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Fast conversion |
| Docling | Free | ⭐⭐⭐⭐ | ⭐⭐⭐ | Advanced conversion |

## 📁 Project Structure
```
ai-document-processor/
├── src/
│   ├── extractors/
│   │   ├── pdf_opensource.py      # PyPDF2, pdfplumber
│   │   ├── pdf_enterprise.py       # Azure Document Intelligence
│   │   └── web_opensource.py       # BeautifulSoup scraper
│   ├── converters/
│   │   ├── docling_converter.py    # Docling integration
│   │   └── markitdown_converter.py # MarkItDown integration
│   ├── storage/
│   │   └── s3_manager.py           # AWS S3 operations
│   └── api/
│       └── main.py                 # FastAPI application
├── streamlit_app/
│   └── app.py                      # Streamlit frontend
├── data/
│   ├── sample_pdfs/                # Sample PDF files
│   └── temp/                       # Temporary processing files
├── docs/
│   └── tool_comparison.md          # Detailed tool comparison
├── tests/                          # Unit tests
├── .env                            # Environment variables (not in git)
├── .gitignore                      # Git ignore rules
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── AiUseDisclosure.md             # AI tool usage disclosure
```

## 👥 Team Contributions

- **Member 1 (33%):** PDF extraction, Azure integration, backend API
- **Member 2 (33%):** Web scraping, markdown conversion, S3 storage
- **Member 3 (33%):** Frontend development, testing, documentation

GitHub Issues: [View task breakdown](https://github.com/DarshanGaviyappa/ai-document-processor/issues)

## 🚀 Deployment

### FastAPI Deployment Options
- **Render:** https://render.com (Free tier available)
- **Railway:** https://railway.app
- **AWS EC2/Lambda**

### Streamlit Deployment
- **Streamlit Community Cloud:** https://streamlit.io/cloud (Free)
- Deploy directly from GitHub repository

### Deployment Steps
1. Push code to GitHub
2. Create account on deployment platform
3. Connect GitHub repository
4. Configure environment variables
5. Deploy!

## 📝 License

MIT License - See LICENSE file for details

## 🔗 Links

- **GitHub Repository:** https://github.com/DarshanGaviyappa/ai-document-processor
- **Live Demo (Streamlit):** [Coming soon]
- **API Documentation:** http://localhost:8000/docs (when running locally)
- **Video Demo:** [Coming soon]

## 📞 Contact

For questions or issues, please create an issue on GitHub or contact the team.

---

Built with ❤️ for DAMG 7245 - Big Data Systems & Intelligence Analytics