# Exporting Model Artifacts from Google Colab

After running all training cells in `tokopedia_merged.ipynb` in Google Colab:

1. Upload `colab/save_artifacts.py` to your Colab session.
2. Run the export cell:

```python
%run save_artifacts.py
```

This will create a zip file containing:

```
tokopedia_artifacts/
├── sentiment_model/
│   ├── model.keras
│   ├── tokenizer.json
│   ├── label_encoder.pkl
│   ├── config.json
│   └── metrics.json
└── category_model/
    ├── model.keras
    ├── tokenizer.json
    ├── label_encoder.pkl
    ├── config.json
    └── metrics.json
```

3. Download the zip file.
4. Extract it into the project root (`raisah/`) so that the `models/` directory is populated.

```bash
cd /path/to/raisah
unzip ~/Downloads/tokopedia_artifacts.zip
```

5. Verify the structure:

```bash
ls models/sentiment_model
ls models/category_model
```

> Note: The script tries to auto-detect the trained model variables (`tuned_model`, `bilstm_s`, `bilstm_c`, etc.). If you used different variable names, update `save_artifacts.py` accordingly.
