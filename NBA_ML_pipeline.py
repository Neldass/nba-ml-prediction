import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from nba_api.stats.endpoints import PlayerGameLog
from nba_api.stats.static import players, teams
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Fonction pour récupérer l'ID du joueur
def get_player_id(player_name):
    nba_players = players.get_players()
    for player in nba_players:
        if player["full_name"].lower() == player_name.lower():
            return player["id"]
    return None

# Fonction pour récupérer la correspondance entre abréviations et noms complets des équipes
def get_team_abbreviation_mapping():
    return {team["abbreviation"]: team["full_name"] for team in teams.get_teams()}

# Fonction pour récupérer tous les matchs de la saison pour le joueur
def load_nba_data(player_name):
    player_id = get_player_id(player_name)
    if player_id is None:
        st.error("Joueur non trouvé. Vérifiez le nom.")
        return None
    
    game_log = PlayerGameLog(player_id=player_id, season='2024-25', season_type_all_star='Regular Season').get_data_frames()[0]
    df = game_log.copy()  # Récupère tous les matchs de la saison

    df["Location"] = df["MATCHUP"].apply(lambda x: "Home" if "vs." in x else "Away")
    df["Opponent"] = df["MATCHUP"].apply(lambda x: x.split()[-1])
    
    team_mapping = get_team_abbreviation_mapping()
    df["Opponent_Full"] = df["Opponent"].map(team_mapping)
    
    # Ajouter de nouvelles features
    df["FG%"] = df["FGM"] / df["FGA"]
    df["FT%"] = df["FTM"] / df["FTA"]
    df["Plus-Minus"] = df["PLUS_MINUS"]
    df["Rolling_PTS_Avg"] = df["PTS"].rolling(window=5, min_periods=1).mean()
    df["Hot_Streak"] = (df["PTS"] > df["PTS"].rolling(window=5, min_periods=1).mean() * 1.2).astype(int)

    # Remplacer les valeurs infinies par NaN puis les remplir
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    
    return df

# Fonction pour l'analyse exploratoire
def exploratory_analysis(df):
    st.write("### Aperçu des données")
    st.dataframe(df)  # Affiche un aperçu des données
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    stats_to_plot = ['PTS', 'REB', 'AST', 'BLK', 'FG3M', 'TOV', 'FG%', 'FT%', 'Plus-Minus']
    for ax, stat in zip(axes.flatten(), stats_to_plot):
        sns.histplot(df[stat], kde=True, ax=ax)
        ax.set_title(f"Distribution de {stat}")
    st.pyplot(fig)

# Fonction de prétraitement
def preprocess_data(df):
    features = ['REB', 'AST', 'BLK', 'FG3M', 'TOV', 'MIN', 'Location', 'Opponent', 'FG%', 'FT%', 'Plus-Minus', 'Rolling_PTS_Avg', 'Hot_Streak']
    target = ['PTS', 'REB', 'AST', 'BLK', 'FG3M']
    
    label_encoder = LabelEncoder()
    df["Location"] = label_encoder.fit_transform(df["Location"])
    df["Opponent"] = label_encoder.fit_transform(df["Opponent"])
    
    missing_features = [col for col in features if col not in df.columns]
    if missing_features:
        st.error(f"Colonnes manquantes dans le DataFrame : {missing_features}")
        return None, None, None, None

    df[features] = df[features].fillna(0)
    imputer = SimpleImputer(strategy="mean")
    df_features = pd.DataFrame(imputer.fit_transform(df[features]), columns=df[features].columns, index=df.index)
    
    X = df_features
    y = df[target]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# Fonctions de visualisation
def plot_mae(results):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=list(results.keys()), y=[r['mae'] for r in results.values()], ax=ax)
    ax.set_title("Comparaison des modèles - Mean Absolute Error (MAE)")
    ax.set_ylabel("MAE (Erreur Absolue Moyenne)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)

def plot_mse(results):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=list(results.keys()), y=[r['mse'] for r in results.values()], ax=ax)
    ax.set_title("Comparaison des modèles - Mean Squared Error (MSE)")
    ax.set_ylabel("MSE (Erreur Quadratique Moyenne)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)


# Entraînement des modèles avec GridSearchCV
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "KNN": KNeighborsRegressor(),
        "SVM": MultiOutputRegressor(SVR()),  
        "Random Forest": RandomForestRegressor(),
        "AdaBoost": MultiOutputRegressor(AdaBoostRegressor()),  
        "Gradient Boosting": MultiOutputRegressor(GradientBoostingRegressor())  
    }
    
    param_grid = {
        "Ridge": {"alpha": [0.1, 1, 10]},
        "Lasso": {"alpha": [0.1, 1, 10]},
        "KNN": {"n_neighbors": [3, 5, 7]},
        "SVM": {"estimator__C": [0.1, 1, 10], "estimator__kernel": ["linear", "rbf"]},  
        "Random Forest": {"n_estimators": [50, 100, 200]},
        "AdaBoost": {"estimator__n_estimators": [50, 100]},  
        "Gradient Boosting": {"estimator__n_estimators": [50, 100]}  
    }
    
    results = {}

    for name, model in models.items():
        st.write(f"Optimisation de {name} avec GridSearchCV...")
        grid_search = GridSearchCV(model, param_grid.get(name, {}), cv=3, scoring="r2")
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results[name] = {
            "r2": r2,
            "mae": mae,
            "mse": mse,
            "params": grid_search.best_params_
        }
        
        st.write(f"{name}: R²={r2:.4f}, MAE={mae:.4f}, MSE={mse:.4f}")
        

    plot_mae(results)
    plot_mse(results)

    # Visualisation des résultats
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=list(results.keys()), y=[r['r2'] for r in results.values()], ax=ax)
    ax.set_title("Comparaison des modèles - Score R² après optimisation")
    ax.set_ylabel("R² Score")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)

# Interface Streamlit
st.title("Prédictions NBA avec Machine Learning")
player_name = st.text_input("Entrez le nom complet du joueur (ex: LeBron James)")

if player_name:
    df = load_nba_data(player_name)
    if df is not None:
        exploratory_analysis(df)
        X_train, X_test, y_train, y_test = preprocess_data(df)
        if X_train is not None:
            train_and_evaluate_models(X_train, X_test, y_train, y_test)