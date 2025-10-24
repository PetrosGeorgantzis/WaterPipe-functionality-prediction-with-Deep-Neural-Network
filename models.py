# ---------------------------------------------------------------------------
# FILE: models.py
"""
Model training and utilities.
Supports: MLP (Keras), RandomForest, XGBoost (if available).
"""
import os
import joblib
import numpy as np

# sklearn models
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# try imports for tensorflow and xgboost lazily
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False


# ---- MLP (Keras) ---------------------------------------------------------

def train_mlp(X: np.ndarray, y: np.ndarray, input_dim: int=None, epochs=50, patience=3):
    if not TF_AVAILABLE:
        raise RuntimeError('TensorFlow is not available. Install tensorflow to train MLP.')
    if input_dim is None:
        input_dim = X.shape[1]

    y = np.array(y)

    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(256, activation='sigmoid')(inputs)
    x = tf.keras.layers.Dense(128, activation='sigmoid')(x)
    x = tf.keras.layers.Dense(64, activation='sigmoid')(x)
    x = tf.keras.layers.Dense(32, activation='sigmoid')(x)
    x = tf.keras.layers.Dense(64, activation='sigmoid')(x)
    x = tf.keras.layers.Dense(128, activation='sigmoid')(x)
    x = tf.keras.layers.Dense(256, activation='sigmoid')(x)
    outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience)
    model.fit(X, y, epochs=epochs, callbacks=[callback], verbose=1)
    return model


# ---- RandomForest --------------------------------------------------------

def train_random_forest(X: np.ndarray, y_labels: np.ndarray, n_estimators=200, random_state=42):
    # y_labels should be 1D integer labels (0..2)
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    rf.fit(X, y_labels)
    return rf


# ---- XGBoost ----------------------------------------------------------------

def train_xgboost(X: np.ndarray, y_labels: np.ndarray, num_round=100, params=None):
    if not XGB_AVAILABLE:
        raise RuntimeError('xgboost not installed. Install via `pip install xgboost`')
    if params is None:
        params = {'objective': 'multi:softprob', 'num_class': 3, 'eval_metric': 'mlogloss'}
    dtrain = xgb.DMatrix(X, label=y_labels)
    bst = xgb.train(params, dtrain, num_round)
    return bst


# ---- Utilities -----------------------------------------------------------

def predict_mlp(model, X: np.ndarray):
    if TF_AVAILABLE:
        return model.predict(X)
    raise RuntimeError('TensorFlow not available')


def predict_rf(model, X: np.ndarray):
    probs = model.predict_proba(X)
    return probs


def predict_xgb(model, X: np.ndarray):
    if not XGB_AVAILABLE:
        raise RuntimeError('xgboost not available')
    d = xgb.DMatrix(X)
    return model.predict(d)


# Save/Load helper
def save_model(obj, path: str):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    if hasattr(obj, 'save') and callable(getattr(obj, 'save')) and path.endswith('.h5'):
        obj.save(path)
    else:
        joblib.dump(obj, path)


def load_model(path: str):
    if path.endswith('.h5'):
        if not TF_AVAILABLE:
            raise RuntimeError('TensorFlow required to load .h5 Keras model')
        return tf.keras.models.load_model(path)
    return joblib.load(path)


# End of models.py