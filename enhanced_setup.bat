@echo off
echo ========================================
echo Enhanced University Chatbot Setup
echo ========================================
echo.

echo Installing enhanced dependencies...
pip install -r requirements.txt

echo.
echo Processing enhanced knowledge base...
python process_pdfs.py

echo.
echo Testing enhanced features...
python test_enhanced_features.py

echo.
echo Starting enhanced chatbot...
echo.
echo The enhanced chatbot now includes:
echo - Intelligent semantic search
echo - Dynamic response generation  
echo - Enhanced chunk quality
echo - University domain classification
echo - Smart fallback mechanisms
echo.
echo Press any key to start the application...
pause >nul

streamlit run app.py 