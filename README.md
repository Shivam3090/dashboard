# Customer Churn & Retention Analytics Dashboard

**Python | Dash | Plotly | Pandas | scikit-learn | Lifelines**

An interactive customer analytics application exploring telecom churn patterns, customer segments, services, and tenure.

## Business problem
How can a subscription business understand churn behavior and identify groups that may need retention attention?

## Features
- Churn visualizations by tenure and contract type
- Customer segmentation with K-Means
- Random Forest classification with confusion matrix and classification report
- Kaplan-Meier survival analysis
- Service distribution and demographic summary tables

## Technical workflow
1. Load the included telecom CSV and clean charges.
2. Engineer service-count and spending features.
3. Explore customer behavior with Plotly.
4. Train clustering and classification models.
5. Display interactive dashboards and model diagnostics.

## Run locally
```bash
git clone https://github.com/Shivam3090/dashboard.git
cd dashboard
python -m venv .venv
pip install -r requirements.txt
python app.py
```

> **Project status:** Portfolio demonstration, not a production churn prediction service. The model uses a train/test split; production use would require additional validation, monitoring, and security controls.

## Next steps
Add automated tests, reproducible model evaluations, a cleaner project structure, and deployment configuration.

[Explore more projects](https://github.com/Shivam3090)
