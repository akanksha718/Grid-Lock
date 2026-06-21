import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# =====================================================
# 1. LOAD DATA
# =====================================================
print("Loading data...")
train_df = pd.read_csv('dataset/train.csv')
test_df = pd.read_csv('dataset/test.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"\nTrain columns: {train_df.columns.tolist()}")
print(f"Test columns: {test_df.columns.tolist()}")
print(f"\nTrain info:")
print(train_df.info())
print(f"\nMissing values in train:")
print(train_df.isnull().sum())
print(f"\nMissing values in test:")
print(test_df.isnull().sum())

# =====================================================
# 2. DATA PREPROCESSING & FEATURE ENGINEERING
# =====================================================
print("\n\nPreprocessing and engineering features...")

def preprocess_data(df, is_test=False):
    """Preprocess and engineer features"""
    df = df.copy()
    
    # Extract time components
    df['hour'] = df['timestamp'].str.split(':').str[0].astype(int)
    df['minute'] = df['timestamp'].str.split(':').str[1].astype(int)
    
    # Handle missing RoadType - fill with mode or 'Unknown'
    if df['RoadType'].isnull().sum() > 0:
        mode_road = train_df['RoadType'].mode()[0] if not is_test else 'Residential'
        df['RoadType'] = df['RoadType'].fillna(mode_road)
    
    # Handle missing Temperature - fill with mean
    if df['Temperature'].isnull().sum() > 0:
        temp_mean = train_df['Temperature'].mean()
        df['Temperature'] = df['Temperature'].fillna(temp_mean)
    
    # Encode categorical variables
    df['LargeVehicles_encoded'] = (df['LargeVehicles'] == 'Allowed').astype(int)
    df['Landmarks_encoded'] = (df['Landmarks'] == 'Yes').astype(int)
    
    # Label encode RoadType
    le_road = LabelEncoder()
    df['RoadType_encoded'] = le_road.fit_transform(df['RoadType'])
    
    # Label encode Weather
    le_weather = LabelEncoder()
    df['Weather_encoded'] = le_weather.fit_transform(df['Weather'].astype(str))
    
    # Encode geohash (use first few characters as location groups)
    df['geohash_prefix'] = df['geohash'].str[:3]
    le_geohash = LabelEncoder()
    df['geohash_encoded'] = le_geohash.fit_transform(df['geohash'])
    
    return df

train_processed = preprocess_data(train_df)
test_processed = preprocess_data(test_df, is_test=True)

print("Features engineered successfully!")
print(f"Train processed shape: {train_processed.shape}")
print(f"Test processed shape: {test_processed.shape}")

# =====================================================
# 3. PREPARE FEATURES FOR MODELING
# =====================================================
print("\nPreparing feature sets...")

# Select features for modeling
feature_cols = [
    'geohash_encoded', 'day', 'hour', 'minute',
    'NumberofLanes', 'LargeVehicles_encoded', 'Landmarks_encoded',
    'Temperature', 'RoadType_encoded', 'Weather_encoded'
]

X_train = train_processed[feature_cols].copy()
y_train = train_processed['demand'].copy()
X_test = test_processed[feature_cols].copy()

# Handle any remaining NaN values
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_train.mean())

print(f"Training features shape: {X_train.shape}")
print(f"Test features shape: {X_test.shape}")
print(f"Target shape: {y_train.shape}")

# =====================================================
# 4. TRAIN MULTIPLE MODELS & SELECT BEST
# =====================================================
print("\n\nTraining models with cross-validation...")

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Model 1: XGBoost
print("\n1. Training XGBoost...")
xgb_model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, 
                         subsample=0.8, random_state=42, n_jobs=-1)
xgb_scores = cross_val_score(xgb_model, X_train, y_train, cv=kfold, 
                             scoring='r2', n_jobs=-1)
print(f"   XGBoost CV R² scores: {xgb_scores}")
print(f"   Mean R²: {xgb_scores.mean():.6f} (+/- {xgb_scores.std():.6f})")

# Model 2: LightGBM
print("\n2. Training LightGBM...")
lgb_model = LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1,
                          num_leaves=31, random_state=42, n_jobs=-1, verbose=-1)
lgb_scores = cross_val_score(lgb_model, X_train, y_train, cv=kfold, 
                             scoring='r2', n_jobs=-1)
print(f"   LightGBM CV R² scores: {lgb_scores}")
print(f"   Mean R²: {lgb_scores.mean():.6f} (+/- {lgb_scores.std():.6f})")

# Model 3: Random Forest
print("\n3. Training Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf_scores = cross_val_score(rf_model, X_train, y_train, cv=kfold, 
                            scoring='r2', n_jobs=-1)
print(f"   Random Forest CV R² scores: {rf_scores}")
print(f"   Mean R²: {rf_scores.mean():.6f} (+/- {rf_scores.std():.6f})")

# =====================================================
# 5. SELECT BEST MODEL & TRAIN ON FULL DATA
# =====================================================
models_scores = {
    'XGBoost': xgb_scores.mean(),
    'LightGBM': lgb_scores.mean(),
    'Random Forest': rf_scores.mean()
}

best_model_name = max(models_scores, key=models_scores.get)
best_score = models_scores[best_model_name]

print(f"\n\nBest Model: {best_model_name} with R² = {best_score:.6f}")

# Train best model on full training data
if best_model_name == 'XGBoost':
    best_model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, 
                             subsample=0.8, random_state=42, n_jobs=-1)
elif best_model_name == 'LightGBM':
    best_model = LGBMRegressor(n_estimators=100, max_depth=6, learning_rate=0.1,
                              num_leaves=31, random_state=42, n_jobs=-1, verbose=-1)
else:
    best_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)

print(f"\nTraining {best_model_name} on full training data...")
best_model.fit(X_train, y_train)

# Validate on training set
train_pred = best_model.predict(X_train)
train_r2 = r2_score(y_train, train_pred)
print(f"Training R² score: {train_r2:.6f}")

# =====================================================
# 6. MAKE PREDICTIONS ON TEST SET
# =====================================================
print("\n\nGenerating predictions on test set...")
predictions = best_model.predict(X_test)

# Ensure predictions are within reasonable bounds
predictions = np.clip(predictions, 0, None)

print(f"Predictions shape: {predictions.shape}")
print(f"Predictions range: [{predictions.min():.6f}, {predictions.max():.6f}]")
print(f"Predictions mean: {predictions.mean():.6f}")
print(f"Sample predictions (first 10): {predictions[:10]}")

# =====================================================
# 7. CREATE SUBMISSION FILE
# =====================================================
print("\n\nCreating submission file...")

submission_df = pd.DataFrame({
    'Index': range(len(predictions)),
    'demand': predictions
})

print(f"\nSubmission shape: {submission_df.shape}")
print(f"Submission columns: {submission_df.columns.tolist()}")
print(f"\nFirst few rows:")
print(submission_df.head())

# Save to CSV
submission_df.to_csv('dataset/submission.csv', index=False)
print(f"\n✓ Submission saved to 'dataset/submission.csv'")
print(f"  File size: {submission_df.shape[0]} rows × {submission_df.shape[1]} columns")

print("\n" + "="*60)
print("SOLUTION COMPLETE!")
print("="*60)
print(f"Model: {best_model_name}")
print(f"CV R² Score: {best_score:.6f}")
print(f"Training R² Score: {train_r2:.6f}")
print(f"Submission file: dataset/submission.csv")
print(f"Expected Score (max(0, 100 × {best_score:.4f})): {max(0, 100 * best_score):.2f}")
