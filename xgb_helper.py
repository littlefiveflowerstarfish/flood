import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np

def custom_obj(preds, dtrain):
    """
    Custom objective for XGBoost binary classification with group softmax.
    
    The raw predictions (preds) are a flat vector of length N,
    where N is a multiple of 730. We reshape to (-1,730), apply softmax
    along axis=1, then “reverse sigmoid” to recover logits and compute
    binary logloss.
    
    Returns:
      grad: gradient (a vector of length N)
      hess: hessian (a vector of length N)
    """
    labels = dtrain.get_label()  # shape (N,)
    N = preds.shape[0]
    ng = N // 730              # number of groups
    # Reshape into (ng,730)
    X = preds.reshape(ng, 730)

    # Compute row–wise softmax. (Subtract max for numerical stability.)
    X_max = np.max(X, axis=1, keepdims=True)
    expX = np.exp(X - X_max)
    prob = expX / np.sum(expX, axis=1, keepdims=True)  # shape (ng,730)
    q = prob.ravel()  # flatten back to shape (N,)
    
    # For each element, define
    #   g = - y/q + (1-y)/(1-q)
    # which is the derivative of the binary logloss (after “reverse sigmoid”)
    g = - (labels / q) + ((1 - labels) / (1 - q))
    
    # For each group, compute Q = sum_j q_j * g_j.
    g_group = (prob * g.reshape(ng, 730)).sum(axis=1, keepdims=True)  # shape (ng,1)
    Q = np.repeat(g_group, 730, axis=1).ravel()
    
    # The gradient w.r.t. each raw prediction is:
    #   grad = q * (g - Q)
    grad = q * (g - Q)
    
    # For the hessian we use the approximation coming from the fact that 
    # d²(loss)/dz² = q*(1-q) (where z = log(q/(1-q)) is the recovered logit)
    # and the (diagonal) derivative of z with respect to x is 1.
    hess = q * (1 - q)
    
    return grad, hess

import numpy as np
import xgboost as xgb

def custom_loss(preds, dtrain):
    labels = dtrain.get_label()
    preds = preds.astype(np.float64)  # Ensure float64 for numerical stability
    N = preds.shape[0]
    M = N // 730
    # Reshape predictions to (M, 730)
    F = preds.reshape(M, 730)
    
    # Compute softmax
    max_F = np.max(F, axis=1, keepdims=True)
    exp_F = np.exp(F - max_F)  # For numerical stability
    sum_exp_F = np.sum(exp_F, axis=1, keepdims=True)
    S = exp_F / sum_exp_F
    
    # Compute gradient of loss with respect to S
    s = S.ravel()
    eps = 1e-8  # To avoid division by zero
    dL_ds = (-labels / (s + eps)) + ((1 - labels) / (1 - s + eps))
    dL_dS = dL_ds.reshape(M, 730)
    
    # Compute gradient of loss with respect to F (raw preds)
    grad_F = np.empty_like(F)
    for m in range(M):
        S_m = S[m]
        dL_dS_m = dL_dS[m]
        sum_term = np.dot(S_m, dL_dS_m)
        grad_F[m] = S_m * (dL_dS_m - sum_term)
    grad = grad_F.ravel().astype(np.float32)
    
    # Compute Hessian approximation (diagonal terms)
    # Second derivative of loss w.r.t. s
    d2L_ds2 = (labels / (s**2 + eps)) + ((1 - labels) / ((1 - s)**2 + eps))
    d2L_ds2 = d2L_ds2.reshape(M, 730)
    
    hess_F = np.empty_like(F)
    for m in range(M):
        S_m = S[m]
        hess_F[m] = S_m * (d2L_ds2[m] * S_m * (1 - S_m) + (1 - 2 * S_m) * (dL_dS[m] - np.dot(S_m, dL_dS[m])))
    hess = hess_F.ravel().astype(np.float32)
    
    return grad, hess

# Example usage:
# dtrain = xgb.DMatrix(X_train, label=y_train)
# params = {
#     'objective': 'binary:logistic',
#     'tree_method': 'hist',
#     'learning_rate': 0.1,
# }
# model = xgb.train(params, dtrain, num_boost_round=10, obj=custom_loss)

def logloss_eval(preds, dtrain):
    """
    Custom evaluation metric for binary logloss.
    Note: Since we are using a custom objective, the raw predictions are the model's
    outputs (logits). Therefore, we need to apply the sigmoid function to get probabilities.
    """
    labels = dtrain.get_label()
    # Transform logits to probabilities using the logistic function
    preds_prob = 1.0 / (1.0 + np.exp(-preds))
    # Clip probabilities for numerical stability
    eps = 1e-15
    preds_prob = np.clip(preds_prob, eps, 1 - eps)
    loss = -np.mean(labels * np.log(preds_prob) + (1 - labels) * np.log(1 - preds_prob))
    return 'logloss', loss

class XGBHelper:
    def __init__(self, task, params=None, num_boost_rounds=100, early_stop_rounds=None, seeds=None):
        """
        XGBoost helper class for classification and regression tasks.
        
        Parameters:
            task (str): 'classification' or 'regression'
            params (dict): XGBoost parameters
            num_boost_round (int): Number of boosting rounds
            early_stop_rounds (int): Early stopping rounds
            seeds (list): List of random seeds for seed ensemble
        """
        self.task = task
        self.params = params.copy() if params else {}
        self.num_boost_rounds = num_boost_rounds
        self.early_stop_rounds = early_stop_rounds
        self.seeds = seeds if seeds else [42]  # Default to a single seed if not provided
        self.models = []  # Store models trained with different seeds
        self.best_iterations = []

        # Set GPU if available and not specified in params
        if 'tree_method' not in self.params:
            self.params['tree_method'] = 'gpu_hist'
        # self.params['objective'] = None
        # self.params['eval_metric'] ='logloss'
        # Set default objective if not provided
        if 'objective' not in self.params:
            if self.task == 'classification':
                self.params['objective'] = 'binary:logistic'
            elif self.task == 'regression':
                self.params['objective'] = 'reg:squarederror'
            else:
                raise ValueError("Task must be 'classification' or 'regression'")

    def fit(self, X, y, X_val=None, y_val=None, tr_margin=None, va_margin=None):
        """
        Train the XGBoost model using seed ensemble.
        
        Parameters:
            X (array-like): Training features
            y (array-like): Training target
            X_val (array-like): Validation features (optional)
            y_val (array-like): Validation target (optional)
            tr_margin (array-like): Training base margin (optional)
            va_margin (array-like): Validation base margin (optional)
        """
        # Handle multi-class classification automatically
        if self.task == 'classification':
            y = np.asarray(y)
            num_classes = len(np.unique(y))
            if num_classes > 2:
                if self.params.get('objective') == 'binary:logistic':
                    self.params['objective'] = 'multi:softmax'
                    self.params['num_class'] = num_classes

        # Apply reverse sigmoid for binary classification margins
        if self.task == 'classification' and self.params.get('objective') == 'binary:logistic':
            def reverse_sigmoid(x):
                x = np.asarray(x)  # Ensure x is a NumPy array
                with np.errstate(divide='ignore', invalid='ignore'): # Handle potential warnings
                    return np.log(x / (1 - x))

            if tr_margin is not None:
                tr_margin = reverse_sigmoid(tr_margin)
            if va_margin is not None:
                va_margin = reverse_sigmoid(va_margin)
        
        self.models = []  # Reset models list
        self.best_iterations = []

        for seed in self.seeds:
            # Set the random seed in the parameters
            self.params['random_state'] = seed

            # Create DMatrix objects
            dtrain = xgb.DMatrix(X, label=y, base_margin=tr_margin)
            evals = [(dtrain, 'train')]

            if X_val is not None and y_val is not None:
                dval = xgb.DMatrix(X_val, label=y_val, base_margin=va_margin)
                evals.append((dval, 'val'))

            # Handle early stopping
            early_stopping = None
            if self.early_stop_rounds and X_val is not None and y_val is not None:
                early_stopping = self.early_stop_rounds

            model = xgb.train(
                params=self.params,
                dtrain=dtrain,
                num_boost_round=self.num_boost_rounds,
                evals=evals,
                # obj=custom_loss,
                # feval=logloss_eval,   # Use our custom evaluation function for logloss
                early_stopping_rounds=early_stopping,
                verbose_eval=100
            )

            self.models.append(model)

            # Save the best iteration if validation set is provided
            if X_val is not None and y_val is not None:
                self.best_iterations.append(model.best_iteration)
            else:
                self.best_iterations.append(None) # Store None if no validation set

        return self

    def predict(self, X, te_margin=None):
        """Make predictions using the seed ensemble, averaging predictions from each model."""
        if not self.models:
            raise RuntimeError("Models not trained. Call fit() first.")

        # Apply reverse sigmoid for binary classification margins
        if self.task == 'classification' and self.params.get('objective') == 'binary:logistic':
            def reverse_sigmoid(x):
                x = np.asarray(x) # Ensure x is a NumPy array
                with np.errstate(divide='ignore', invalid='ignore'): # Handle potential warnings
                    return np.log(x / (1 - x))
            if te_margin is not None:
                te_margin = reverse_sigmoid(te_margin)

        dtest = xgb.DMatrix(X, base_margin=te_margin)
        predictions = []

        for model, best_iteration in zip(self.models, self.best_iterations):
            if best_iteration is not None:
                pred = model.predict(dtest, iteration_range=(0, best_iteration + 1))
            else:
                pred = model.predict(dtest)

            predictions.append(pred)

        # Average the predictions across all models
        avg_prediction = np.mean(predictions, axis=0)

        # If it's a classification task, and objective is set to multi:softmax, convert to class labels
        # if self.task == 'classification' and self.params['objective'] == 'multi:softmax':
        #   avg_prediction = np.argmax(avg_prediction, axis=1)
        
        return avg_prediction

    def get_models(self):
      """Get the list of trained XGBoost models"""
      return self.models

    def plot_feature_importance(self, model_index=0, importance_type='weight', max_num_features=None, **kwargs):
        """
        Plots the feature importance for a specific model in the ensemble.

        Parameters:
            model_index (int): Index of the model in the ensemble to plot feature importance for.
            importance_type (str): 'weight', 'gain', or 'cover'
            max_num_features (int): Maximum number of features to display (default: None - display all)
            **kwargs: Additional keyword arguments to pass to xgb.plot_importance()
        """
        if not self.models:
            raise RuntimeError("Models not trained. Call fit() first.")
        if model_index < 0 or model_index >= len(self.models):
            raise ValueError(f"Invalid model_index. Must be between 0 and {len(self.models) - 1}")
            
        xgb.plot_importance(self.models[model_index], importance_type=importance_type, max_num_features=max_num_features, **kwargs)
        plt.show()

    def get_feature_importance(self, model_index=0, importance_type='weight'):
        """
        Returns the feature importances as a pandas DataFrame for a specific model in the ensemble.

        Parameters:
            model_index (int): Index of the model in the ensemble to get feature importance for.
            importance_type (str): 'weight', 'gain', or 'cover'

        Returns:
            pandas.DataFrame: A DataFrame with feature names as index and importance scores as a column.
        """
        if not self.models:
            raise RuntimeError("Models not trained. Call fit() first.")
        if model_index < 0 or model_index >= len(self.models):
            raise ValueError(f"Invalid model_index. Must be between 0 and {len(self.models) - 1}")

        importance_dict = self.models[model_index].get_score(importance_type=importance_type)
        
        # Convert the dictionary to a pandas DataFrame
        importance_df = pd.DataFrame(list(importance_dict.items()), columns=['Feature', 'Importance'])
        importance_df = importance_df.set_index('Feature').sort_values('Importance', ascending=False)
        
        return importance_df