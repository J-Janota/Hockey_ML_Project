import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

def debug_hockey_model():
    print("=== HOCKEY XG MODEL DEBUGGER ===\n")
    
    df1 = pd.read_csv("shots_2020.csv")
    df2 = pd.read_csv("shots_2021.csv")
    df3 = pd.read_csv("shots_2022.csv")
    df4 = pd.read_csv("shots_2023.csv")
    df5 = pd.read_csv("shots_2024.csv")

    frames = [df1, df2, df3, df4, df5]

    df = pd.concat(frames)

    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['goal'].value_counts()}")
    print(f"Target percentage: {df['goal'].mean():.3f}")
    
    print(f"\nMissing values per column (top 10):")
    missing_counts = df.isnull().sum().sort_values(ascending=False)
    print(missing_counts.head(10))

    all_features = ['arenaAdjustedShotDistance', 'arenaAdjustedXCord', 'arenaAdjustedYCord', 
                   'averageRestDifference', 'awayEmptyNet', 'awayPenalty1Length', 
                   'awayPenalty1TimeLeft', 'awaySkatersOnIce', 'awayTeamCode', 'awayTeamGoals', 
                   'homeEmptyNet', 'homePenalty1Length', 'homePenalty1TimeLeft', 
                   'homeSkatersOnIce', 'homeTeamCode', 'homeTeamGoals', 'isHomeTeam', 
                   'isPlayoffGame', 'lastEventCategory', 'lastEventShotAngle', 
                   'lastEventShotDistance', 'lastEventTeam', 'location', 'offWing', 
                   'period', 'playerPositionThatDidEvent', 'shooterLeftRight', 
                   'shooterPlayerId', 'shotAngleAdjusted', 'shotType', 'shotOnEmptyNet',
                   'shotRush', 'team', 'teamCode', 'xCord', 'xCordAdjusted', 
                   'yCord', 'yCordAdjusted']
    
    leakage_features = [
        'shotWasOnGoal', 'shotGoalieFroze', 'shotGeneratedRebound',
        'shotPlayStopped', 'shotPlayContinuedInZone', 'shotPlayContinuedOutsideZone',
        'shotRebound', 'event'
    ]
    
    available_features = [f for f in all_features if f in df.columns]
    print(f"\nUsing {len(available_features)} features out of {len(all_features)} requested")
    
    df_work = df.copy()
    
    categorical_cols = df_work[available_features].select_dtypes(include=['object', 'category']).columns
    print(f"\nCategorical columns to encode: {list(categorical_cols)}")
    
    for col in categorical_cols:
        unique_vals = df_work[col].nunique()
        if unique_vals > 20:
            print(f"  Label encoding {col} ({unique_vals} categories)")
            le = LabelEncoder()
            df_work[col] = le.fit_transform(df_work[col].fillna('MISSING'))
        else:
            print(f"  One-hot encoding {col} ({unique_vals} categories)")
            dummies = pd.get_dummies(df_work[col], prefix=col, drop_first=True)
            df_work = pd.concat([df_work, dummies], axis=1)
            available_features.extend(dummies.columns.tolist())
            available_features.remove(col)
    
    X = df_work[available_features].copy()
    y = df_work['goal'].copy()
    
    print(f"\nFinal feature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    missing_features = X.columns[X.isnull().any()].tolist()
    if missing_features:
        print(f"\nFeatures with missing values: {len(missing_features)}")
        for col in missing_features[:5]:
            print(f"  {col}: {X[col].isnull().sum()} missing")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train), 
        columns=X_train.columns, 
        index=X_train.index
    )
    X_test_imputed = pd.DataFrame(
        imputer.transform(X_test), 
        columns=X_test.columns, 
        index=X_test.index
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)

    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n=== RESULTS ===")
    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    print(f"\n=== SANITY CHECKS ===")
    print(f"Predicted probability range: {y_pred_proba.min():.4f} to {y_pred_proba.max():.4f}")
    print(f"Mean predicted probability: {y_pred_proba.mean():.4f}")
    print(f"Actual goal rate: {y_test.mean():.4f}")

    pred_counts = pd.Series(y_pred).value_counts()
    print(f"Prediction distribution: {dict(pred_counts)}")

    feature_importance = pd.DataFrame({
        'feature': available_features,
        'importance': np.abs(model.coef_[0])
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 most important features:")
    print(feature_importance.head(10))

    if auc >= 0.99:
        print(f"\n⚠️  WARNING: AUC = {auc:.4f} suggests possible data leakage!")
        print("Check if any features are consequences of the target variable.")
    
    return model, scaler, imputer, available_features, df_work

def predict_sample_shots(model, scaler, imputer, features, df_work, n_shots=20):
    """Predict probabilities for the first n shots and compare with actual outcomes"""
    print(f"\n=== PREDICTING FIRST {n_shots} SHOTS ===")
    correct_sum = 0
    
    for i in range(min(n_shots, len(df_work))):
        shot_data = df_work.iloc[i:i+1][features].copy()
        actual_goal = df_work.iloc[i]['goal']
        
        moneypuck_xg = df_work.iloc[i].get('xGoal', None)
        
        shot_imputed = pd.DataFrame(
            imputer.transform(shot_data), 
            columns=features, 
            index=shot_data.index
        )
        shot_scaled = scaler.transform(shot_imputed)
        
        pred_proba = model.predict_proba(shot_scaled)[0, 1]
        
        print(f"\nShot {i+1}:")
        print(f"  Actual Goal: {'YES' if actual_goal == 1 else 'NO'}")
        print(f"  Our Model xG: {pred_proba:.4f}")
        if moneypuck_xg is not None:
            print(f"  MoneyPuck xG: {moneypuck_xg:.4f}")
            print(f"  Difference: {abs(pred_proba - moneypuck_xg):.4f}")
        if abs(pred_proba - moneypuck_xg) < 0.01:
            print("  MoneyPuck xG matches our model xG!")
            correct_sum += 1
        
        key_features = ['arenaAdjustedShotDistance', 'shotAngleAdjusted', 'shotType', 'period']
        available_key_features = [f for f in key_features if f in shot_data.columns]
        
        if available_key_features:
            print(f"  Key features:")
            for feature in available_key_features:
                value = shot_data.iloc[0][feature]
                print(f"    {feature}: {value}")
    print(f"\nTotal correct matches: {correct_sum} out of {n_shots} shots")

def predict_sample_shots_no_text(model, scaler, imputer, features, df_work, n_shots):
    correct_sum = 0
    for i in range(min(n_shots, len(df_work))):
        shot_data = df_work.iloc[i:i+1][features].copy()
        actual_goal = df_work.iloc[i]['goal']
        
        moneypuck_xg = df_work.iloc[i].get('xGoal', None)
        
        shot_imputed = pd.DataFrame(
            imputer.transform(shot_data), 
            columns=features, 
            index=shot_data.index
        )
        shot_scaled = scaler.transform(shot_imputed)
        
        pred_proba = model.predict_proba(shot_scaled)[0, 1]
        

        if abs(pred_proba - moneypuck_xg) < 0.01:
            correct_sum += 1
        if i % 1000 == 0:
            print(i)
    print(f"\nTotal correct matches: {correct_sum} out of {n_shots} shots")
if __name__ == "__main__":
    model, scaler, imputer, features, df_work = debug_hockey_model()
    predict_sample_shots(model, scaler, imputer, features, df_work, n_shots=20)