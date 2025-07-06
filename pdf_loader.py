import os
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
from typing import List, Dict, Tuple
import logging

# Set up logging
logger = logging.getLogger(__name__)

def load_pdfs_from_directory(directory_path):
    """
    Load all PDFs from a specified directory
    
    Args:
        directory_path: Path to the directory containing PDF files
        
    Returns:
        Dictionary with filename as key and text content as value
    """
    pdf_contents = {}
    
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'rb') as f:
                        pdf_reader = PyPDF2.PdfReader(f)
                        text = ""
                        for page_num in range(len(pdf_reader.pages)):
                            page_text = pdf_reader.pages[page_num].extract_text()
                            # Clean up the extracted text
                            page_text = clean_extracted_text(page_text)
                            text += page_text + "\n"
                        
                        if text.strip():  # Only add if text was extracted
                            pdf_contents[file] = text
                        else:
                            print(f"Warning: No text extracted from {file}")
                except Exception as e:
                    print(f"Error processing {file}: {e}")
    
    return pdf_contents

def clean_extracted_text(text):
    """
    Clean and normalize extracted text from PDFs
    
    Args:
        text: Raw extracted text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove page numbers and headers/footers
    text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)
    
    # Clean up common PDF extraction artifacts
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Fix missing spaces
    text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)  # Fix sentence boundaries
    
    # Remove excessive line breaks
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    return text.strip()

def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    """
    Split text into smaller chunks for processing with enhanced logic
    
    Args:
        text: The text to split
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    # Pre-process text to identify natural break points
    text = preprocess_text_for_chunking(text)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
    )
    
    chunks = text_splitter.split_text(text)
    
    # Post-process chunks to ensure quality
    enhanced_chunks = []
    for chunk in chunks:
        enhanced_chunk = post_process_chunk(chunk)
        if enhanced_chunk and len(enhanced_chunk.strip()) > 50:  # Minimum meaningful length
            enhanced_chunks.append(enhanced_chunk)
    
    return enhanced_chunks

def preprocess_text_for_chunking(text):
    """
    Preprocess text to improve chunking quality
    
    Args:
        text: Raw text
        
    Returns:
        Preprocessed text
    """
    # Identify and preserve important sections
    text = preserve_important_sections(text)
    
    # Normalize formatting
    text = normalize_text_formatting(text)
    
    return text

def preserve_important_sections(text):
    """
    Preserve important sections that should not be split
    
    Args:
        text: Raw text
        
    Returns:
        Text with preserved sections
    """
    # Identify table-like structures and preserve them
    lines = text.split('\n')
    preserved_lines = []
    
    for i, line in enumerate(lines):
        # Check if this line looks like a table header or data
        if is_table_line(line):
            # Try to group with next few lines if they're also table-like
            table_group = [line]
            j = i + 1
            while j < len(lines) and j < i + 5 and is_table_line(lines[j]):
                table_group.append(lines[j])
                j += 1
            
            # Join table lines with special separator
            preserved_lines.append(' |TABLE| '.join(table_group))
            i = j - 1  # Skip the lines we've already processed
        else:
            preserved_lines.append(line)
    
    return '\n'.join(preserved_lines)

def is_table_line(line):
    """
    Check if a line looks like table data
    
    Args:
        line: Text line
        
    Returns:
        True if line appears to be table data
    """
    # Check for common table patterns
    patterns = [
        r'\d+\s+\d+',  # Numbers separated by spaces
        r'[A-Z][a-z]+\s+\d+',  # Words followed by numbers
        r'\$\d+',  # Dollar amounts
        r'\d+\.\d+',  # Decimal numbers
        r'[A-Z]+\s+[A-Z]+',  # All caps words
    ]
    
    for pattern in patterns:
        if re.search(pattern, line):
            return True
    
    return False

def normalize_text_formatting(text):
    """
    Normalize text formatting for better chunking
    
    Args:
        text: Raw text
        
    Returns:
        Normalized text
    """
    # Standardize bullet points
    text = re.sub(r'[•·▪▫◦‣⁃]', '•', text)
    
    # Standardize numbering
    text = re.sub(r'^\d+\.\s*', r'\g<0>', text, flags=re.MULTILINE)
    
    # Clean up spacing around punctuation
    text = re.sub(r'\s+([.!?])', r'\1', text)
    text = re.sub(r'([.!?])\s*', r'\1 ', text)
    
    return text

def post_process_chunk(chunk):
    """
    Post-process individual chunks to improve quality
    
    Args:
        chunk: Raw chunk text
        
    Returns:
        Enhanced chunk text
    """
    if not chunk or len(chunk.strip()) < 10:
        return None
    
    # Clean up the chunk
    chunk = chunk.strip()
    
    # Remove orphaned words at the beginning
    chunk = re.sub(r'^[a-z]\s+', '', chunk)
    
    # Ensure proper sentence endings
    if not chunk.endswith(('.', '!', '?', ':', ';')):
        # Try to find a good ending point
        sentences = re.split(r'[.!?]', chunk)
        if len(sentences) > 1:
            # Remove incomplete last sentence
            chunk = '. '.join(sentences[:-1]) + '.'
    
    # Restore table formatting if present
    if '|TABLE|' in chunk:
        chunk = chunk.replace(' |TABLE| ', '\n')
    
    return chunk

def extract_semantic_sections(text):
    """
    Extract semantic sections from text for better chunking
    
    Args:
        text: Raw text
        
    Returns:
        List of semantic sections
    """
    sections = []
    
    # Split by common section headers
    section_patterns = [
        r'(?:^|\n)([A-Z][A-Z\s]+:?)',  # ALL CAPS headers
        r'(?:^|\n)(\d+\.\s*[A-Z][^:\n]+)',  # Numbered sections
        r'(?:^|\n)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*:)',  # Title case headers
    ]
    
    current_section = ""
    lines = text.split('\n')
    
    for line in lines:
        is_header = any(re.match(pattern, line) for pattern in section_patterns)
        
        if is_header and current_section:
            sections.append(current_section.strip())
            current_section = line
        else:
            current_section += "\n" + line if current_section else line
    
    if current_section:
        sections.append(current_section.strip())
    
    return sections if sections else [text]

def process_pdf_directory(directory_path, chunk_size=800, chunk_overlap=250):
    """
    Process all PDFs in a directory and return chunks with enhanced metadata
    
    Args:
        directory_path: Path to directory with PDFs
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of dictionaries with text chunks and metadata
    """
    pdf_contents = load_pdfs_from_directory(directory_path)
    all_chunks = []
    
    for filename, content in pdf_contents.items():
        # Extract semantic sections first
        sections = extract_semantic_sections(content)
        
        for section_idx, section in enumerate(sections):
            # Split sections into chunks
            chunks = split_text_into_chunks(section, chunk_size, chunk_overlap)
            
            for chunk_idx, chunk in enumerate(chunks):
                # Create enhanced metadata
                metadata = create_enhanced_metadata(filename, section_idx, chunk_idx, chunk)
                
                all_chunks.append({
                    "text": chunk,
                    "metadata": metadata
                })
    
    return all_chunks

def create_enhanced_metadata(filename, section_idx, chunk_idx, chunk_text):
    """
    Create enhanced metadata for chunks
    
    Args:
        filename: Source filename
        section_idx: Section index
        chunk_idx: Chunk index within section
        chunk_text: Chunk text content
        
    Returns:
        Enhanced metadata dictionary
    """
    metadata = {
        "source": filename,
        "section_id": section_idx,
        "chunk_id": chunk_idx,
        "type": "pdf",
        "chunk_length": len(chunk_text),
        "word_count": len(chunk_text.split()),
        "has_numbers": bool(re.search(r'\d+', chunk_text)),
        "has_dates": bool(re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', chunk_text)),
        "has_currency": bool(re.search(r'\$[\d,]+', chunk_text)),
        "has_bullet_points": '•' in chunk_text,
        "has_table_data": '|TABLE|' in chunk_text or is_table_line(chunk_text),
        "content_type": classify_content_type(chunk_text)
    }
    
    return metadata

def classify_content_type(text):
    """
    Classify the content type of a chunk
    
    Args:
        text: Chunk text
        
    Returns:
        Content type classification
    """
    text_lower = text.lower()
    
    # Check for different content types
    if re.search(r'\$\d+', text):
        return 'financial'
    elif re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text):
        return 'temporal'
    elif '•' in text or re.search(r'^\d+\.', text, re.MULTILINE):
        return 'list'
    elif is_table_line(text):
        return 'tabular'
    elif len(text.split('.')) > 3:
        return 'narrative'
    else:
        return 'general' 