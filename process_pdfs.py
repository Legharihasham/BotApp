import os
from pdf_loader import process_pdf_directory
from web_scraper import main as process_web_links
from embeddings_manager import EmbeddingsManager
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Process all PDFs and web links, then generate combined embeddings with enhanced chunking
    """
    logger.info("Starting enhanced PDF and web processing...")
    
    # Process the main PDFs directory with optimized chunk sizes for better context
    pdf_dir = os.path.join("Data", "PDF's")
    print(f"Processing PDFs in {pdf_dir}...")
    # Enhanced chunk size and overlap for better context preservation and semantic understanding
    main_chunks = process_pdf_directory(pdf_dir, chunk_size=600, chunk_overlap=250)
    print(f"Processed {len(main_chunks)} chunks from main PDFs")
    
    # Process the fee structure PDFs - using optimized chunks for fee tables and financial information
    fee_dir = os.path.join("Data", "Fee_structure")
    print(f"Processing PDFs in {fee_dir}...")
    fee_chunks = process_pdf_directory(fee_dir, chunk_size=500, chunk_overlap=200)
    print(f"Processed {len(fee_chunks)} chunks from fee structure PDFs")
    
    # Process web links - web content often needs larger chunks for better context
    links_file = os.path.join("Data", "Links.txt")
    print(f"Processing web links from {links_file}...")
    web_chunks = process_web_links(links_file, chunk_size=700, chunk_overlap=250)
    print(f"Processed {len(web_chunks)} chunks from web links")
    
    # Enhance chunks with better metadata and semantic information
    enhanced_chunks = enhance_chunks_with_metadata(main_chunks + fee_chunks + web_chunks)
    
    # Combine all chunks
    all_chunks = enhanced_chunks
    print(f"Total enhanced chunks: {len(all_chunks)}")
    
    # Create embeddings with enhanced model
    print("Creating embeddings with BGE model...")
    embeddings_manager = EmbeddingsManager(model_name="BAAI/bge-base-en-v1.5")
    embeddings = embeddings_manager.create_embeddings(all_chunks)
    print(f"Created embeddings with shape: {embeddings.shape}")
    
    # Save embeddings
    print("Saving enhanced embeddings...")
    index_path, chunks_path = embeddings_manager.save_embeddings(filename_prefix="university_combined")
    print(f"Saved index to {index_path}")
    print(f"Saved chunks to {chunks_path}")
    
    # Print statistics about the enhanced knowledge base
    print_chunk_statistics(all_chunks)
    
    print("Enhanced processing complete!")

def enhance_chunks_with_metadata(chunks):
    """
    Enhance chunks with better metadata and semantic information
    
    Args:
        chunks: List of raw chunks
        
    Returns:
        List of enhanced chunks with better metadata
    """
    enhanced_chunks = []
    
    for i, chunk in enumerate(chunks):
        enhanced_chunk = chunk.copy()
        
        # Add semantic category based on content
        enhanced_chunk["metadata"]["semantic_category"] = classify_chunk_semantics(chunk["text"])
        
        # Add content quality score
        enhanced_chunk["metadata"]["content_quality"] = assess_content_quality(chunk["text"])
        
        # Add chunk importance score
        enhanced_chunk["metadata"]["importance_score"] = calculate_importance_score(chunk["text"])
        
        # Add enhanced source information
        if "source" in chunk["metadata"]:
            enhanced_chunk["metadata"]["source_type"] = classify_source_type(chunk["metadata"]["source"])
        
        # Add chunk ID for better tracking
        enhanced_chunk["metadata"]["chunk_id"] = f"chunk_{i:06d}"
        
        enhanced_chunks.append(enhanced_chunk)
    
    return enhanced_chunks

def classify_chunk_semantics(text):
    """
    Classify chunk content into semantic categories
    
    Args:
        text: Chunk text content
        
    Returns:
        Semantic category string
    """
    text_lower = text.lower()
    
    # Define semantic categories with keywords
    categories = {
        'academic_program': ['course', 'program', 'degree', 'major', 'minor', 'curriculum', 'syllabus'],
        'admission_process': ['admission', 'enrollment', 'application', 'deadline', 'requirement'],
        'financial_info': ['fee', 'tuition', 'payment', 'cost', 'scholarship', 'financial aid'],
        'campus_facility': ['facility', 'building', 'campus', 'library', 'lab', 'classroom'],
        'student_services': ['service', 'support', 'help', 'assistance', 'guidance'],
        'administrative': ['procedure', 'process', 'policy', 'regulation', 'rule'],
        'general_info': ['information', 'about', 'overview', 'introduction']
    }
    
    # Count matches for each category
    category_scores = {}
    for category, keywords in categories.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        category_scores[category] = score
    
    # Return the category with the highest score
    if category_scores:
        return max(category_scores, key=category_scores.get)
    
    return 'general_info'

def assess_content_quality(text):
    """
    Assess the quality of chunk content
    
    Args:
        text: Chunk text content
        
    Returns:
        Quality score (0-1)
    """
    if not text or len(text.strip()) < 10:
        return 0.1
    
    # Factors that contribute to quality
    factors = {
        'length': min(len(text) / 500, 1.0),  # Optimal length around 500 chars
        'structure': 1.0 if any(char in text for char in ['•', '-', '1.', '2.', '3.']) else 0.5,
        'specificity': 1.0 if any(word in text.lower() for word in ['specific', 'detail', 'information', 'procedure']) else 0.7,
        'completeness': 1.0 if text.count('.') >= 2 else 0.6  # Multiple sentences
    }
    
    # Calculate weighted average
    weights = {'length': 0.3, 'structure': 0.2, 'specificity': 0.3, 'completeness': 0.2}
    quality_score = sum(factors[key] * weights[key] for key in factors)
    
    return min(quality_score, 1.0)

def calculate_importance_score(text):
    """
    Calculate importance score for the chunk
    
    Args:
        text: Chunk text content
        
    Returns:
        Importance score (0-1)
    """
    text_lower = text.lower()
    
    # Important keywords that indicate high-value content
    important_keywords = [
        'fee', 'tuition', 'admission', 'deadline', 'requirement', 'procedure',
        'application', 'registration', 'enrollment', 'scholarship', 'financial aid',
        'course', 'program', 'degree', 'curriculum', 'syllabus'
    ]
    
    # Calculate importance based on keyword presence
    keyword_count = sum(1 for keyword in important_keywords if keyword in text_lower)
    importance_score = min(keyword_count / 5, 1.0)  # Normalize to 0-1
    
    # Boost score for chunks with specific information
    if any(word in text_lower for word in ['specific', 'detail', 'exact', 'precise']):
        importance_score = min(importance_score + 0.2, 1.0)
    
    return importance_score

def classify_source_type(source):
    """
    Classify the source type of the chunk
    
    Args:
        source: Source information
        
    Returns:
        Source type classification
    """
    if not source:
        return 'unknown'
    
    source_lower = source.lower()
    
    if 'pdf' in source_lower or '.pdf' in source_lower:
        return 'pdf_document'
    elif 'web' in source_lower or 'http' in source_lower:
        return 'web_content'
    elif 'fee' in source_lower or 'payment' in source_lower:
        return 'financial_document'
    elif 'admission' in source_lower or 'application' in source_lower:
        return 'admission_document'
    else:
        return 'general_document'

def print_chunk_statistics(chunks):
    """
    Print statistics about the processed chunks
    
    Args:
        chunks: List of processed chunks
    """
    print("\n=== CHUNK STATISTICS ===")
    print(f"Total chunks: {len(chunks)}")
    
    # Count by type
    type_counts = {}
    category_counts = {}
    quality_scores = []
    importance_scores = []
    
    for chunk in chunks:
        # Count by type
        chunk_type = chunk["metadata"].get("type", "unknown")
        type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
        
        # Count by semantic category
        category = chunk["metadata"].get("semantic_category", "unknown")
        category_counts[category] = category_counts.get(category, 0) + 1
        
        # Collect quality and importance scores
        quality_scores.append(chunk["metadata"].get("content_quality", 0))
        importance_scores.append(chunk["metadata"].get("importance_score", 0))
    
    print(f"\nChunks by type:")
    for chunk_type, count in type_counts.items():
        print(f"  {chunk_type}: {count}")
    
    print(f"\nChunks by semantic category:")
    for category, count in category_counts.items():
        print(f"  {category}: {count}")
    
    print(f"\nQuality scores - Average: {sum(quality_scores)/len(quality_scores):.2f}")
    print(f"Importance scores - Average: {sum(importance_scores)/len(importance_scores):.2f}")
    
    # Show high-quality chunks
    high_quality_chunks = [chunk for chunk in chunks if chunk["metadata"].get("content_quality", 0) > 0.8]
    print(f"\nHigh-quality chunks (>0.8): {len(high_quality_chunks)}")
    
    # Show high-importance chunks
    high_importance_chunks = [chunk for chunk in chunks if chunk["metadata"].get("importance_score", 0) > 0.8]
    print(f"High-importance chunks (>0.8): {len(high_importance_chunks)}")

if __name__ == "__main__":
    main() 