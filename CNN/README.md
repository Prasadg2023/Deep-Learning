# CNN MNIST Streamlit App

Quick steps to run the app locally (Windows PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
streamlit run cnn.py
```

Notes:
- If you have a GPU and want GPU-enabled TensorFlow, install the appropriate `tensorflow` build for your system.
- If `import keras` errors appear, ensure you're using the `tensorflow` package (the app uses `tensorflow.keras`).
