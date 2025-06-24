# 🏀 NBA Player Performance Prediction with Machine Learning

This project predicts NBA player performance (PTS, REB, AST...) for upcoming games using supervised machine learning models.

## 📌 Project Description

Developed as part of the "Expert in Data Science" program at Sigma Clermont, this project analyzes player stats and match context using models like Ridge, Lasso, SVR, and ensemble methods.

## 📊 Dataset

- Source: Real-time data via [nba_api](https://github.com/swar/nba_api)
- Features: Player stats, match location, opponent, shooting precision, momentum metrics.
- Targets: Points, rebounds, assists, blocks, 3-point shots made.

## 🔍 Models Used

- Ridge Regression
- Lasso Regression
- Support Vector Regression
- K-Nearest Neighbors
- Random Forest
- AdaBoost
- Gradient Boosting

## 🛠️ Technologies

- Python (pandas, scikit-learn, numpy, seaborn, matplotlib)
- Streamlit for interactive predictions
- nba_api for data extraction

## 🚀 How to Run

```bash
pip install -r requirements.txt
streamlit run streamlit_app/NBA_ML_pipeline.py
