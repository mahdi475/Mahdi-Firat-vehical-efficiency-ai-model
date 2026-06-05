# TAM400 MPG Prediction Project

This project predicts vehicle fuel efficiency (MPG) using the Auto MPG dataset from the UCI Machine Learning Repository.

## Objective

The goal is to apply a complete machine learning workflow: data loading, preprocessing, exploratory data analysis, model training, evaluation, and interpretation.

## Dataset

The dataset used is the Auto MPG dataset from the UCI Machine Learning Repository.

Target variable:
- mpg

Features:
- cylinders
- displacement
- horsepower
- weight
- acceleration
- model_year
- origin

## Models

The following models were tested:
- Dummy Regressor
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

## Best Result

The best model was Random Forest Regressor.

Approximate test performance:
- MAE: 1.58
- RMSE: 2.16
- R2: 0.91

## Repository Structure

```text
TAM400-MPG-Project/
├── README.md
├── notebook.ipynb
├── requirements.txt
├── data/
│   └── auto-mpg.data
├── figures/
│   ├── mpg_distribution.png
│   ├── weight_vs_mpg.png
│   ├── horsepower_vs_mpg.png
│   └── model_results.png
└── presentation/
    └── TAM400_MPG_Project_Presentation_Mahdi.pptx
```

## Tools

- Python
- pandas
- NumPy
- matplotlib
- scikit-learn
- Jupyter Notebook

## AI Acknowledgment

Generative AI tools such as ChatGPT were used to support understanding, debugging, explanation improvement, and presentation preparation. All code, results, and conclusions were reviewed and understood by the students.
