# EMHMD - Health Misinformation Detector

A Streamlit app for detecting health misinformation using BioBERT and a TF-IDF baseline.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Render

1. Push this project to GitHub.
2. In Render, create a new Web Service.
3. Connect the GitHub repository.
4. Use these settings:
   - Build command: `pip install -r requirements.txt`
   - Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

## Notes

- The app will work with the base BioBERT model even if the fine-tuned weight file is not present.
- Large model files are ignored by git to keep the repository lightweight.
