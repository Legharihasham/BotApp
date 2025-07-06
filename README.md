# 🎓 Enhanced University Chatbot (GRAIN AI)

An intelligent, dynamic university assistant chatbot that provides comprehensive information about university procedures, courses, fees, and campus life. Built with advanced semantic search and AI-powered response generation.

## ✨ Enhanced Features

### 🧠 Intelligent Semantic Search
- **Advanced Query Enhancement**: Automatically enhances user queries with related university terms for better semantic understanding
- **Dynamic Threshold Filtering**: Uses different relevance thresholds based on query type and context
- **Smart Fallback Mechanisms**: When exact matches aren't found, intelligently searches with broader criteria
- **University-Specific Keywords**: Built-in knowledge of academic, administrative, financial, and campus-related terms

### 🎯 Dynamic Response Generation
- **Domain-Aware Responses**: Classifies queries into university domains (academic, administrative, financial, etc.)
- **General Knowledge Integration**: Provides helpful responses even when specific information isn't available
- **Context Enhancement**: Automatically adds relevant general university knowledge to responses
- **Natural Conversation Flow**: Responds like a knowledgeable university advisor, not a robotic AI

### 📚 Enhanced Knowledge Base
- **Intelligent Chunking**: Creates semantic chunks that preserve context and meaning
- **Quality Assessment**: Automatically evaluates and scores chunk quality and importance
- **Metadata Enrichment**: Rich metadata including content type, semantic category, and relevance scores
- **Table and List Preservation**: Special handling for structured data like fee tables and lists

### 🔍 Advanced Search Capabilities
- **Multi-Strategy Search**: Combines semantic similarity with keyword-based search
- **Relevance Scoring**: Sophisticated scoring system that considers multiple factors
- **Duplicate Elimination**: Removes redundant chunks while preserving the best information
- **Source Type Filtering**: Can search across PDF documents, web content, or both

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up Environment
Create a `.env` file with your Groq API key:
```
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Process Knowledge Base
```bash
python process_pdfs.py
```

### 4. Run the Application
```bash
streamlit run app.py
```

## 🛠️ Technical Architecture

### Enhanced Components

#### 1. **EmbeddingsManager** (`embeddings_manager.py`)
- **University Keyword Classification**: Built-in knowledge of academic, administrative, financial, campus, and student life domains
- **Query Enhancement**: Automatically adds related terms to improve search results
- **Smart Filtering**: Different filtering strategies based on query type
- **Broad Search Fallback**: When initial search yields limited results, performs keyword-based searches

#### 2. **GeminiAPI** (`gemini_api.py`)
- **Domain Classification**: Automatically classifies queries into university domains
- **Dynamic Response Generation**: Provides helpful responses even with limited context
- **General Knowledge Integration**: Adds relevant university knowledge when specific information isn't available
- **Enhanced Prompting**: More sophisticated prompt engineering for better responses

#### 3. **PDF Loader** (`pdf_loader.py`)
- **Semantic Section Extraction**: Identifies and preserves logical document sections
- **Table and List Preservation**: Special handling for structured data
- **Content Quality Assessment**: Evaluates chunk quality and importance
- **Enhanced Metadata**: Rich metadata for better search and filtering

#### 4. **Process PDFs** (`process_pdfs.py`)
- **Enhanced Chunking Strategy**: Optimized chunk sizes for different content types
- **Quality Scoring**: Automatic assessment of chunk quality and importance
- **Statistical Analysis**: Provides detailed statistics about the knowledge base

## 🎯 Key Improvements

### 1. **More Intelligent Responses**
- No more "I don't have enough information" responses
- Provides general university guidance when specific details aren't available
- Suggests where to find more information

### 2. **Better Context Understanding**
- Understands university-specific terminology and concepts
- Provides relevant responses even with partial information
- Maintains conversation context across multiple queries

### 3. **Enhanced Search Quality**
- More accurate chunk retrieval
- Better handling of related topics
- Improved relevance scoring

### 4. **Dynamic Response Generation**
- Adapts responses based on available context
- Provides helpful information even with limited knowledge base coverage
- More natural, conversational responses

## 🔧 Configuration Options

### Debug Mode
Enable debug mode in the sidebar to see:
- Relevance scores and filtering reasons
- Chunk metadata and quality scores
- Search strategy details
- Session memory information

### Data Source Selection
Choose which data sources to use:
- **All Sources**: Combines PDF and web content
- **PDF Documents Only**: Uses only official university documents
- **Website Content Only**: Uses only web-scraped content

### Session Memory
- Enable to maintain conversation context
- View conversation history with `/history` command
- Automatic conversation summarization for long sessions

## 📊 Knowledge Base Statistics

The enhanced processing provides detailed statistics including:
- Total chunks processed
- Chunks by source type (PDF, web)
- Chunks by semantic category
- Quality and importance scores
- High-quality chunk identification

## 🎓 University Domains Supported

The chatbot understands and can respond to queries about:

### Academic
- Courses and programs
- Degree requirements
- Curriculum and syllabi
- Academic planning

### Administrative
- Admissions procedures
- Registration and enrollment
- Application deadlines
- Administrative processes

### Financial
- Tuition and fees
- Payment procedures
- Scholarships and financial aid
- Cost structures

### Campus
- Facilities and buildings
- Libraries and labs
- Housing options
- Campus infrastructure

### Student Life
- Student activities
- Clubs and organizations
- Campus events
- Student services

## 🔍 Search Capabilities

### Semantic Search
- Uses BGE embeddings for semantic understanding
- Enhanced query processing with related terms
- Multi-strategy search with fallback mechanisms

### Keyword Search
- University-specific keyword classification
- Domain-aware search strategies
- Intelligent keyword expansion

### Context-Aware Filtering
- Different relevance thresholds for different query types
- Quality-based chunk selection
- Duplicate elimination and ranking

## 🚀 Performance Optimizations

- **Efficient Embedding Storage**: FAISS index for fast similarity search
- **Smart Caching**: Session-based memory and context caching
- **Optimized Chunking**: Semantic-aware text splitting
- **Quality Filtering**: Automatic removal of low-quality chunks

## 📝 Usage Examples

### Basic Queries
```
"What are the admission requirements?"
"How much are the tuition fees?"
"What courses are available?"
```

### Complex Queries
```
"I'm interested in computer science programs, what should I know about the curriculum and fees?"
"What's the process for international student admission and housing?"
"Can you tell me about student life and campus facilities?"
```

### Follow-up Queries
```
"What about the application deadline?"
"Are there any scholarships available?"
"What documents do I need to submit?"
```

## 🔧 Troubleshooting

### Common Issues

1. **No Knowledge Base Found**
   - Run `python process_pdfs.py` first
   - Ensure PDF files are in the correct directories

2. **API Key Issues**
   - Check your `.env` file has the correct `GROQ_API_KEY`
   - Verify the API key is valid and has sufficient credits

3. **Poor Response Quality**
   - Enable debug mode to see search results
   - Check chunk quality scores in the debug information
   - Consider reprocessing the knowledge base

### Performance Tips

1. **For Better Responses**
   - Use specific, detailed questions
   - Enable session memory for follow-up questions
   - Check debug mode to understand search results

2. **For Faster Processing**
   - Use specific data sources (PDF only or web only)
   - Adjust relevance threshold in debug mode
   - Clear session history if it gets too long

## 🤝 Contributing

To contribute to the enhancement of this chatbot:

1. **Improve Chunking**: Enhance the semantic section extraction in `pdf_loader.py`
2. **Add Domains**: Extend university domains in `gemini_api.py`
3. **Enhance Search**: Improve search strategies in `embeddings_manager.py`
4. **Better Prompts**: Refine the prompt engineering in `gemini_api.py`

## 📄 License

This project is designed for educational and university use. Please ensure compliance with your institution's policies and data privacy requirements.

---

**Built for students. Not Sci-fi.** 🎓
