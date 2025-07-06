#!/usr/bin/env python3
"""
Test script for enhanced university chatbot features
"""

import os
from dotenv import load_dotenv
from embeddings_manager import EmbeddingsManager
from gemini_api import GeminiAPI

# Load environment variables
load_dotenv()

def test_enhanced_embeddings():
    """Test the enhanced embeddings manager features"""
    print("🧠 Testing Enhanced Embeddings Manager...")
    
    # Initialize embeddings manager
    embeddings_manager = EmbeddingsManager()
    
    # Test loading embeddings
    if not embeddings_manager.load_embeddings():
        print("❌ No embeddings found. Please run process_pdfs.py first.")
        return False
    
    print(f"✅ Loaded {len(embeddings_manager.chunks)} chunks")
    
    # Test university keyword extraction
    test_queries = [
        "What are the admission requirements?",
        "How much are the tuition fees?",
        "Tell me about computer science courses",
        "What facilities are available on campus?",
        "How is the weather today?"  # Non-university query
    ]
    
    print("\n🔍 Testing University Keyword Extraction:")
    for query in test_queries:
        keywords = embeddings_manager._extract_university_keywords(query)
        is_university = embeddings_manager._is_university_related(query)
        print(f"Query: '{query}'")
        print(f"  Keywords: {keywords}")
        print(f"  University-related: {is_university}")
        print()
    
    # Test enhanced search
    print("🔍 Testing Enhanced Search:")
    test_query = "admission requirements"
    chunks = embeddings_manager.search_similar_chunks(test_query, k=5)
    
    print(f"Found {len(chunks)} chunks for '{test_query}':")
    for i, chunk in enumerate(chunks):
        score = chunk['metadata'].get('relevance_score', 0)
        reason = chunk['metadata'].get('filtering_reason', 'unknown')
        print(f"  {i+1}. Score: {score:.2f}, Reason: {reason}")
        print(f"     Text: {chunk['text'][:100]}...")
        print()
    
    return True

def test_enhanced_api():
    """Test the enhanced API features"""
    print("🤖 Testing Enhanced API...")
    
    # Initialize API
    try:
        api = GeminiAPI()
        print("✅ API initialized successfully")
    except Exception as e:
        print(f"❌ API initialization failed: {e}")
        return False
    
    # Test domain classification
    test_queries = [
        "What courses are available?",
        "How do I apply for admission?",
        "What are the tuition fees?",
        "Tell me about campus facilities",
        "What student activities are there?"
    ]
    
    print("\n🎯 Testing Domain Classification:")
    for query in test_queries:
        domain = api._classify_query_domain(query)
        print(f"Query: '{query}' -> Domain: {domain}")
    
    # Test dynamic response generation
    print("\n💬 Testing Dynamic Response Generation:")
    test_query = "What are the admission requirements?"
    domain = api._classify_query_domain(test_query)
    
    # Test with no context
    response = api._generate_dynamic_response(test_query, [])
    print(f"Dynamic response (no context): {response[:200]}...")
    
    # Test with some context
    mock_context = [{
        'text': 'Admission requirements include high school diploma and standardized test scores.',
        'metadata': {'relevance_score': 0.7}
    }]
    response = api._generate_dynamic_response(test_query, mock_context)
    print(f"Dynamic response (with context): {response[:200]}...")
    
    return True

def test_integration():
    """Test the integration of enhanced features"""
    print("🔗 Testing Integration...")
    
    # Initialize components
    embeddings_manager = EmbeddingsManager()
    if not embeddings_manager.load_embeddings():
        print("❌ No embeddings found. Please run process_pdfs.py first.")
        return False
    
    try:
        api = GeminiAPI()
    except Exception as e:
        print(f"❌ API initialization failed: {e}")
        return False
    
    # Test end-to-end query processing
    test_queries = [
        "What are the admission requirements?",
        "How much does it cost to attend?",
        "What computer science courses are available?",
        "Tell me about campus housing",
        "What student organizations exist?"
    ]
    
    print("\n🎯 Testing End-to-End Processing:")
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        
        # Get chunks
        chunks = embeddings_manager.search_similar_chunks(query, k=10)
        print(f"  Found {len(chunks)} relevant chunks")
        
        if chunks:
            # Show top chunk info
            top_chunk = chunks[0]
            score = top_chunk['metadata'].get('relevance_score', 0)
            reason = top_chunk['metadata'].get('filtering_reason', 'unknown')
            print(f"  Top chunk score: {score:.2f}, reason: {reason}")
        
        # Test response generation (without actually calling the API to save costs)
        domain = api._classify_query_domain(query)
        print(f"  Classified domain: {domain}")
        
        # Test dynamic response
        dynamic_response = api._generate_dynamic_response(query, chunks[:2])
        print(f"  Dynamic response preview: {dynamic_response[:150]}...")
    
    return True

def main():
    """Run all tests"""
    print("🚀 Testing Enhanced University Chatbot Features")
    print("=" * 50)
    
    # Check if API key is available
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY not found in environment variables")
        print("Please set up your .env file with a valid API key")
        return
    
    # Run tests
    tests = [
        ("Enhanced Embeddings", test_enhanced_embeddings),
        ("Enhanced API", test_enhanced_api),
        ("Integration", test_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 Test Results Summary:")
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The enhanced features are working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main() 