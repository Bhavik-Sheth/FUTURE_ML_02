# Implementation Plan for Churn Prediction Enhancements

This plan outlines the steps to add the missing features to `Future_Interns_ML_02.ipynb`.

## 1. Detailed Evaluation Metrics & Confusion Matrix
**Goal:** Go beyond simple accuracy to understand how well the model identifies churners (Recall/Precision).

*   **Action:** Import `classification_report` and `confusion_matrix` from `sklearn.metrics`.
*   **Implementation:**
    ```python
    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    # Generate a detailed report
    print("--- Random Forest Evaluation Report ---")
    print(classification_report(y_test, y_pred_forest))

    # Visualizing the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_forest)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=forest.classes_)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix - Random Forest")
    plt.show()
    ```

## 2. Feature Importance Chart
**Goal:** Visualize which factors (e.g., Contract Type, Monthly Charges) contribute most to customer churn.

*   **Action:** Extract `feature_importances_` from the trained Random Forest model and plot them.
*   **Implementation:**
    ```python
    import numpy as np

    # Get feature importances
    importances = forest.feature_importances_
    feature_names = x_train.columns

    # Sort them for better visualization
    indices = np.argsort(importances)[::-1]

    # Plot
    plt.figure(figsize=(12, 6))
    plt.title("Feature Importance (Drivers of Churn)")
    plt.bar(range(x_train.shape[1]), importances[indices], align="center")
    plt.xticks(range(x_train.shape[1]), feature_names[indices], rotation=90)
    plt.tight_layout()
    plt.show()
    ```

## 3. Churn Probability Prediction
**Goal:** Instead of just a "Yes/No" prediction, provide a probability score (e.g., "85% risk of churn").

*   **Action:** Use the `.predict_proba()` method instead of `.predict()`.
*   **Implementation:**
    ```python
    # Get probabilities for the positive class (Churn = 1)
    y_probs = forest.predict_proba(x_test)[:, 1]

    # Create a summary DataFrame
    results_df = pd.DataFrame({
        'Actual_Churn': y_test.iloc[:,0].values,
        'Predicted_Class': y_pred_forest,
        'Churn_Probability': y_probs
    })

    # Display high-risk customers
    print(results_df.head(10))
    ```

## 4. Business Insights Report (Dashboard/PDF)
**Goal:** Summarize findings for stakeholders.

*   **Action:** Since a full dashboard (like Streamlit) requires a separate script, we can generate a static summary within the notebook or export plots.
*   **Implementation:**
    *   Save the Feature Importance plot as an image: `plt.savefig('feature_importance.png')`.
    *   Save the Confusion Matrix as an image: `plt.savefig('confusion_matrix.png')`.
    *   Export the probability predictions to CSV: `results_df.to_csv('churn_predictions.csv', index=False)`.
