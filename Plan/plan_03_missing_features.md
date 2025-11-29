# Missing Features & Implementation Plan

Based on your checklist, here is the status of your current project:

*   ✅ **Explore and clean a customer dataset** (Done in notebook)
*   ✅ **Engineer relevant features** (Done via `get_dummies`)
*   ✅ **Train and test classification models** (Logistic, RF, XGBoost implemented)
*   ✅ **Evaluate model accuracy, precision, recall, and churn probabilities** (Code added in previous step)
*   ✅ **Visualize key churn drivers** (Feature Importance plot added)
*   ❌ **Present your results in a business-focused dashboard or PDF report** (Currently, we only save separate images/CSVs. A unified report is missing.)

---

## Implementation Plan for Missing Features

To fulfill the final requirement, you can implement either a **Unified PDF Report** or an **Interactive Dashboard**.

### Option 1: Unified PDF Report (Recommended for Notebooks)
Instead of saving separate PNG files, generate a single professional PDF containing all insights.

**Implementation Strategy:**
Use `matplotlib.backends.backend_pdf` to save multiple plots to a single file.

```python
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

# Create a PDF object
with PdfPages('Churn_Analysis_Report.pdf') as pdf:
    
    # Page 1: Title & Metrics
    plt.figure(figsize=(8.5, 11))
    plt.axis('off')
    plt.text(0.5, 0.9, "Customer Churn Analysis Report", ha='center', fontsize=24, fontweight='bold')
    plt.text(0.1, 0.8, f"Random Forest Accuracy: {score_forest:.2%}", fontsize=14)
    plt.text(0.1, 0.75, "Key Insight: See subsequent pages for drivers and confusion matrix.", fontsize=12)
    pdf.savefig()
    plt.close()

    # Page 2: Feature Importance (Re-plotting for PDF)
    plt.figure(figsize=(10, 6))
    indices = np.argsort(importances)[::-1]
    plt.title("Top Churn Drivers (Feature Importance)")
    plt.bar(range(20), importances[indices][:20], align="center") # Top 20
    plt.xticks(range(20), feature_names[indices][:20], rotation=90)
    plt.tight_layout()
    pdf.savefig()  # Save the current figure
    plt.close()

    # Page 3: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_forest)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=forest.classes_)
    disp.plot(cmap='Blues')
    plt.title("Model Confusion Matrix")
    pdf.savefig()
    plt.close()

print("PDF Report generated: Churn_Analysis_Report.pdf")
```

### Option 2: Interactive Dashboard (Streamlit)
For a more modern approach, create a separate Python script (`app.py`) to run a web dashboard.

**Implementation Strategy:**
1.  **Save your model** in the notebook:
    ```python
    import joblib
    joblib.dump(forest, 'churn_model.pkl')
    ```
2.  **Create `app.py`** (New File):
    ```python
    import streamlit as st
    import pandas as pd
    import joblib
    import matplotlib.pyplot as plt

    # Load Model & Data
    model = joblib.load('churn_model.pkl')
    data = pd.read_csv('churn_predictions.csv')

    st.title("📊 Customer Churn Dashboard")

    # KPI Metrics
    st.metric("Total Customers Analyzed", len(data))
    st.metric("Predicted Churn Rate", f"{data['Predicted_Class'].mean():.1%}")

    # Visualizations
    st.subheader("High Risk Customers")
    st.dataframe(data[data['Churn_Probability'] > 0.8].head())

    st.subheader("Churn Probability Distribution")
    st.bar_chart(data['Churn_Probability'])
    ```
3.  **Run in Terminal:** `streamlit run app.py`
