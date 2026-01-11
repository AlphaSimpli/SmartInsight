# SmartInsight (PyTorch) — End-to-End Churn Analytics

This repository contains a single notebook that builds a full churn analytics pipeline: data loading, EDA, preprocessing, PyTorch training, evaluation, explainability (Captum), and executive reporting (rule-based + optional LLM).

## What's inside
- `SmartInsight.ipynb`: the complete end-to-end notebook.
- `dataset.csv`: sample dataset used by the notebook (expects a `churn` target column).

## Notebook workflow
The notebook walks through:
1. Data loading and exploratory analysis
2. Preprocessing (one-hot encoding, missing values, scaling)
3. PyTorch model training and evaluation
4. Saving model + preprocessing artifacts
5. Inference on new data
6. Explainability with Integrated Gradients (Captum)
7. Rule-based insights and JSON payload generation
8. Executive summary report (template-based)
9. Optional LLM report (Hugging Face or other provider)

## Quick start
You can run the notebook locally or in Google Colab.

### Option A: Google Colab (recommended)
1. Upload `SmartInsight.ipynb` and `dataset.csv` to Colab.
2. Update the dataset path in the notebook:
   ```python
   DATA_PATH = "/content/dataset.csv"
   ```
3. Run all cells in order.

### Option B: Local run (Jupyter)
Install dependencies:
```bash
pip install pandas numpy torch scikit-learn matplotlib captum joblib
```
Then open the notebook:
```bash
jupyter notebook SmartInsight.ipynb
```
Update the dataset path in the notebook to your local file location.

## Data requirements
The notebook expects a CSV with a binary churn target:
- Target column: `churn` (0/1)
- It also references `customer_id` and `signup_date` in preprocessing. If your dataset does not contain them, adjust the notebook accordingly.

## Outputs (saved files)
When you run the notebook end-to-end, it saves:
- `churn_model.pt`: PyTorch model weights
- `preprocess.pkl`: scaler + feature names
- `insight_payload.json`: structured insights
- `executive_summary.txt`: rule-based summary
- `executive_report.md`: template-based report
- `llm_executive_report.md` / `llm_executive_report_v2.md`: optional LLM reports

## Important security note
The notebook includes an example of setting `HF_TOKEN` for Hugging Face. Do NOT commit real tokens to GitHub. Use environment variables or Colab secrets and remove any hardcoded tokens before publishing.

## License
Add a license that fits your intended usage (e.g., MIT, Apache-2.0).
