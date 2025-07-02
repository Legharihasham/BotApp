@echo off
echo -------------------------------
echo University Assistant Bot Update
echo -------------------------------
echo.
echo Step 1: Installing requirements...
pip install -r requirements.txt

echo.
echo Step 2: Processing data with improved chunking and embeddings...
python process_pdfs.py

echo.
echo Step 3: Starting Streamlit application...
streamlit run app.py 