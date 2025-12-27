"""
Streamlit Frontend for AI Document Processor
"""
import streamlit as st
import requests
import json
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="AI Document Processor",
    page_icon="📄",
    layout="wide"
)

# API URL
API_URL = "http://localhost:8000"

# Title
st.title("🤖 AI Document Processor")
st.markdown("Extract, convert, and store documents using AI-powered tools")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    upload_to_s3 = st.checkbox("Upload to S3", value=True)
    st.markdown("---")
    st.markdown("### 📊 Statistics")
    
    # Check API health
    try:
        health_response = requests.get(f"{API_URL}/health")
        if health_response.status_code == 200:
            st.success("✅ API Online")
        else:
            st.error("❌ API Offline")
    except:
        st.error("❌ API Offline")

# Main content - Tabs
tab1, tab2, tab3 = st.tabs(["📄 PDF Upload", "🌐 Web Scraping", "📚 Results"])

# Tab 1: PDF Upload
with tab1:
    st.header("📄 PDF Document Processing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        method = st.selectbox(
            "Extraction Method",
            ["opensource", "enterprise"],
            help="Open-source: PyPDF2/pdfplumber | Enterprise: Azure Document Intelligence"
        )
    
    with col2:
        converter = st.selectbox(
            "Markdown Converter",
            ["markitdown", "docling"],
            help="Tool to convert extracted content to Markdown"
        )
    
    uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
    
    if uploaded_file and st.button("🚀 Process PDF", type="primary"):
        with st.spinner("Processing PDF..."):
            try:
                # Prepare files and data
                files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                data = {
                    "method": method,
                    "converter": converter,
                    "upload_to_s3": str(upload_to_s3).lower()
                }
                
                # Make API request
                response = requests.post(
                    f"{API_URL}/process-pdf",
                    files=files,
                    data=data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.success("✅ PDF processed successfully!")
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Pages Extracted", result["extraction"]["pages"])
                    
                    with col2:
                        st.metric("Tables Found", result["extraction"]["tables"])
                    
                    with col3:
                        st.metric("Text Length", result["extraction"]["text_length"])
                    
                    # S3 Upload Info
                    if result.get("s3_upload") and result["s3_upload"]["success"]:
                        st.info(f"📦 Uploaded to S3: `{result['s3_upload']['s3_key']}`")
                        st.code(result['s3_upload']['s3_url'], language=None)
                    
                    # Display Extracted Markdown Content
                    st.markdown("---")
                    st.subheader("📝 Extracted Content (Markdown)")
                    
                    # Read the markdown file
                    markdown_file = result.get("markdown_file")
                    if markdown_file:
                        try:
                            with open(markdown_file, 'r', encoding='utf-8') as f:
                                markdown_content = f.read()
                            
                            # Show in tabs
                            content_tab1, content_tab2 = st.tabs(["📄 Rendered", "💻 Raw Markdown"])
                            
                            with content_tab1:
                                st.markdown(markdown_content)
                            
                            with content_tab2:
                                st.code(markdown_content, language='markdown')
                            
                            # Download button
                            st.download_button(
                                label="⬇️ Download Markdown",
                                data=markdown_content,
                                file_name=f"extracted_{uploaded_file.name}.md",
                                mime="text/markdown"
                            )
                        except Exception as e:
                            st.warning(f"Could not read markdown file: {e}")
                    
                    # Show full result in expander
                    with st.expander("📋 View Full API Response"):
                        st.json(result)
                else:
                    st.error(f"❌ Error: {response.text}")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Tab 2: Web Scraping
with tab2:
    st.header("🌐 Webpage Processing")
    
    url = st.text_input(
        "Enter Webpage URL",
        placeholder="https://example.com/article"
    )
    
    converter_web = st.selectbox(
        "Markdown Converter",
        ["markitdown", "docling"],
        key="web_converter",
        help="Tool to convert scraped content to Markdown"
    )
    
    if url and st.button("🚀 Scrape Webpage", type="primary"):
        with st.spinner("Scraping webpage..."):
            try:
                # Prepare data
                data = {
                    "url": url,
                    "converter": converter_web,
                    "upload_to_s3": str(upload_to_s3).lower()
                }
                
                # Make API request
                response = requests.post(
                    f"{API_URL}/process-webpage",
                    data=data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.success("✅ Webpage scraped successfully!")
                    
                    # Display results
                    st.subheader(f"📰 {result['scraping']['title']}")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Text Length", result["scraping"]["text_length"])
                    
                    with col2:
                        st.metric("Headings", result["scraping"]["headings"])
                    
                    with col3:
                        st.metric("Links", result["scraping"]["links"])
                    
                    with col4:
                        st.metric("Images", result["scraping"]["images"])
                    
                    # S3 Upload Info
                    if result.get("s3_upload") and result["s3_upload"]["success"]:
                        st.info(f"📦 Uploaded to S3: `{result['s3_upload']['s3_key']}`")
                        st.code(result['s3_upload']['s3_url'], language=None)
                    
                    # Display Extracted Markdown Content
                    st.markdown("---")
                    st.subheader("📝 Scraped Content (Markdown)")
                    
                    # Read the markdown file
                    markdown_file = result.get("markdown_file")
                    if markdown_file:
                        try:
                            with open(markdown_file, 'r', encoding='utf-8') as f:
                                markdown_content = f.read()
                            
                            # Show in tabs
                            content_tab1, content_tab2 = st.tabs(["📄 Rendered", "💻 Raw Markdown"])
                            
                            with content_tab1:
                                st.markdown(markdown_content[:5000])  # First 5000 chars
                                if len(markdown_content) > 5000:
                                    st.info("Content truncated for display. Download full version below.")
                            
                            with content_tab2:
                                st.code(markdown_content[:5000], language='markdown')
                                if len(markdown_content) > 5000:
                                    st.info("Content truncated for display. Download full version below.")
                            
                            # Download button
                            st.download_button(
                                label="⬇️ Download Markdown",
                                data=markdown_content,
                                file_name=f"scraped_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                                mime="text/markdown"
                            )
                        except Exception as e:
                            st.warning(f"Could not read markdown file: {e}")
                    
                    # Show full result in expander
                    with st.expander("📋 View Full API Response"):
                        st.json(result)
                else:
                    st.error(f"❌ Error: {response.text}")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Tab 3: Results History
with tab3:
    st.header("📚 Processing Results")
    st.info("This section shows your S3 bucket contents")
    
    if st.button("🔄 List S3 Files"):
        try:
            import boto3
            import os
            from dotenv import load_dotenv
            
            load_dotenv()
            
            s3_client = boto3.client(
                's3',
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                region_name=os.getenv('AWS_REGION', 'us-east-1')
            )
            
            bucket_name = os.getenv('S3_BUCKET_NAME')
            
            response = s3_client.list_objects_v2(Bucket=bucket_name)
            
            if 'Contents' in response:
                st.success(f"Found {len(response['Contents'])} files in S3")
                
                for obj in response['Contents']:
                    with st.expander(f"📄 {obj['Key']}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Size:** {obj['Size']} bytes")
                        with col2:
                            st.write(f"**Modified:** {obj['LastModified']}")
                        st.code(f"s3://{bucket_name}/{obj['Key']}", language=None)
            else:
                st.info("No files found in S3 bucket")
                
        except Exception as e:
            st.error(f"Error listing S3 files: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using FastAPI + Streamlit | 
    <a href='http://localhost:8000/docs' target='_blank'>API Docs</a>
    </p>
</div>
""", unsafe_allow_html=True)