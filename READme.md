# California Housing Price Prediction

## 📌 Project Overview

This project uses machine learning to predict median house values in California districts based on demographic, geographic, and housing-related features.

The project follows a practical machine learning workflow, from data exploration and preprocessing to model training, evaluation, and hyperparameter tuning.

**Objective:** Build and compare regression models to identify an effective approach for predicting California housing prices.

## 🎯 Objectives

- Explore and understand a real-world housing dataset.
- Identify relationships between features and house prices.
- Prepare data for machine learning.
- Train and compare multiple regression models.
- Evaluate models using appropriate regression metrics.
- Investigate prediction errors and potential overfitting.
- Improve performance through feature engineering and hyperparameter tuning.

## 📊 Dataset

The project uses the California Housing dataset provided by Scikit-learn.

- **Total observations:** 20,640
- **Input features:** 8
- **Target variable:** `MedHouseVal`
- **Problem type:** Supervised learning — regression

### Features

| Feature      | Description                              |
| ------------ | ---------------------------------------- |
| `MedInc`     | Median income in the district            |
| `HouseAge`   | Median house age                         |
| `AveRooms`   | Average number of rooms per household    |
| `AveBedrms`  | Average number of bedrooms per household |
| `Population` | District population                      |
| `AveOccup`   | Average household occupancy              |
| `Latitude`   | Geographic latitude                      |
| `Longitude`  | Geographic longitude                     |

The target, `MedHouseVal`, represents median house value in units of $100,000. For example, a target value of `2.5` represents $250,000.

**Note:** The dataset's target values are capped at 5.0, corresponding to $500,000.

## 🛠️ Technologies Used

- **Python** — programming language
- **Pandas** — data manipulation and analysis
- **NumPy** — numerical computing
- **Matplotlib** — data visualization
- **Seaborn** — statistical visualization
- **Scikit-learn** — model training and evaluation
- **Jupyter Notebook** — interactive experimentation

## 🔍 Project Workflow

### 1. Data Loading and Exploration

Loaded the California Housing dataset and examined its structure, statistical summaries, feature distributions, and relationships.

### 2. Data Quality Checks

- Checked for missing values.
- Checked for duplicate observations.
- Inspected data types and feature distributions.

### 3. Exploratory Data Analysis

Investigated correlations and visual relationships between housing prices and their features.

Key observations included:

- Median income showed a strong positive relationship with median house value.
- Geographic location was associated with differences in house prices.
- Relationships between individual features and house prices were not always linear.

### 4. Data Preparation

Separated the input features from the target variable and split the dataset into training and testing sets.

- **Training set:** 16,512 observations
- **Testing set:** 4,128 observations
- **Test size:** 20%
- **Random state:** 42

### 5. Model Training

Trained and compared the following regression approaches:

1. Dummy Regressor — baseline model
2. Linear Regression
3. Decision Tree Regressor
4. Random Forest Regressor
5. Histogram-based Gradient Boosting Regressor

### 6. Model Evaluation

Used the following regression metrics:

- **Mean Absolute Error (MAE):** Measures the average absolute difference between actual and predicted values.
- **Mean Squared Error (MSE):** Measures the average squared prediction error.
- **Root Mean Squared Error (RMSE):** Expresses the square root of the average squared error in the target's units.
- **R² Score:** Measures predictive performance relative to a mean-prediction baseline.

### 7. Residual Analysis

Examined the differences between actual and predicted house values.

The analysis indicated that the Random Forest model tended to overpredict lower-valued houses and underpredict higher-valued houses. Prediction errors also increased across higher price ranges.

### 8. Feature Importance and Engineering

Examined feature importance from the Random Forest model.

Median income was the most influential feature, accounting for approximately 56% of the model's reported feature importance.

Tested an additional bedroom-to-room ratio feature. It did not meaningfully improve Random Forest performance.

### 9. Hyperparameter Tuning

Experimented with Random Forest parameters, including maximum tree depth, minimum samples per leaf, and number of trees.

Also compared different iteration counts for Histogram-based Gradient Boosting to investigate performance improvements and diminishing returns.

## 📈 Model Performance

The following results were obtained during experimentation.

| Model                                 |    Test R² |   Test MAE |
| ------------------------------------- | ---------: | ---------: |
| Dummy Regressor                       |    -0.0002 |     0.9061 |
| Linear Regression                     |     0.5758 |     0.5332 |
| Decision Tree Regressor               |     0.6205 |     0.5008 |
| Random Forest Regressor               |     0.7920 |     0.3456 |
| **Histogram-based Gradient Boosting** | **0.8481** | **0.2941** |

_Values are rounded. Results may vary with different dataset splits or model configurations._

### Best Model

The best-performing model among the configurations tested was `HistGradientBoostingRegressor`.

Configuration:

```python
HistGradientBoostingRegressor(
    max_iter=500,
    learning_rate=0.1,
    max_leaf_nodes=31,
    random_state=42
)
```

### Final Results

- **Test R²:** 0.8481
- **Test MAE:** 0.2941
- **Approximate MAE in dollars:** $29,410

The R² score indicates that the model explains approximately 84.81% of the variation in house values in the test set, relative to a mean-prediction baseline.

The MAE corresponds to an average absolute prediction error of approximately $29,410.

These metrics describe performance across the test set; they do not guarantee the accuracy of individual predictions.

## 🧠 Key Learnings

Through this project, I practiced:

- Loading and exploring real-world datasets.
- Performing exploratory data analysis.
- Preparing data for supervised learning.
- Establishing a baseline before training more complex models.
- Comparing linear and tree-based regression approaches.
- Evaluating models using multiple metrics.
- Understanding overfitting and generalization.
- Analyzing residuals to identify systematic prediction errors.
- Interpreting feature importance.
- Testing feature engineering ideas.
- Tuning hyperparameters and comparing model performance.

## ⚠️ Limitations

- The target variable is capped at $500,000, which limits predictions for higher-valued properties.
- Prediction errors vary across house-price ranges.
- Model performance depends on the dataset and the selected configuration.
- Repeated use of the test set during model selection may make the reported performance optimistic. Cross-validation and a final evaluation on an untouched test set would provide a more rigorous assessment.
- The model is intended for educational purposes and should not be treated as a professional property valuation tool.

## 🚀 Future Improvements

Potential next steps include:

- Using cross-validation for more rigorous model selection.
- Investigating alternative hyperparameters.
- Exploring additional feature engineering.
- Saving the trained model for reuse.
- Creating a simple prediction interface.
- Deploying the model through an API or web application.

Adjust the paths and filenames to match your actual repository. The dataset is loaded directly through Scikit-learn, so you do not need a separate dataset file unless you choose to save one.

## ▶️ Getting Started

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

Replace the placeholders with your repository details.

### 2. Create a virtual environment (optional but recommended)

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

Install the libraries used by your notebook:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the project notebook and run the cells in order.

## 📚 Dataset Source

The dataset is available through Scikit-learn's `fetch_california_housing` utility.

Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html

## 👩‍💻 Author

Developed as part of my practical machine learning learning journey through AI_WALKS.

This project represents my second end-to-end regression project, focusing on data exploration, model comparison, evaluation, and improvement.
