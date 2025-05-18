import datetime
import uuid
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F # Often used with nn.Module
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split # Likely used in _split_dataset
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Any, List, Tuple, Optional # For better type hinting
import logging
import random # Add random import
import pandas as pd
import joblib # Added joblib import
try:
    from prophet import Prophet
except ImportError:
    Prophet = None
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path # Add Path import if not already present at top of file

logger = logging.getLogger(__name__)

# --- Placeholder for external service/repository classes ---
# These would typically be defined elsewhere and imported.
# Adding basic stubs here for NetworkMLAnalytics to be syntactically complete regarding its __init__ args.

class BaseServicePlaceholder:
    def __init__(self, name: str):
        self.name = name
        logger.info(f"{self.name} (placeholder) initialized.")

class TelemetryService(BaseServicePlaceholder):
    def __init__(self):
        super().__init__("TelemetryService")
    # Define methods like get_current_telemetry as needed

class MetricStore(BaseServicePlaceholder):
    def __init__(self):
        super().__init__("MetricStore")
    async def get_metrics(self, metrics: List[str], start_time: datetime.datetime, end_time: datetime.datetime, resolution: str) -> Dict[str, Any]:
        logger.info(f"MetricStore: Getting metrics {metrics} from {start_time} to {end_time} with {resolution} resolution.")
        # Simulate fetching data, e.g., for a 1-hour window with 1-minute resolution = 60 points
        num_points = max(10, int((end_time - start_time).total_seconds() / 60)) # Approximation for points based on 1min resolution
        
        # Generate timestamps
        timestamps = [start_time + datetime.timedelta(minutes=i) for i in range(num_points)]
        if not timestamps:
             timestamps = [start_time] # Ensure at least one timestamp if duration is very short
        num_points = len(timestamps)

        data = {"timestamps": timestamps}
        for metric_name in metrics:
            # Simulate some pattern or just random data
            if "error" in metric_name.lower():
                values = [random.uniform(0, 1e-5) for _ in range(num_points)]
            elif "power" in metric_name.lower():
                values = [random.uniform(-5, 5) for _ in range(num_points)] # dBm
            elif "snr" in metric_name.lower():
                values = [random.uniform(15, 35) for _ in range(num_points)] # dB
            else:
                values = [random.random() * 100 for _ in range(num_points)]
            
            # Simulate device IDs, perhaps one or a few different ones
            possible_device_ids = [f"device_{(i%3)+1}" for i in range(num_points)] 
            random.shuffle(possible_device_ids)

            data[metric_name] = {
                "values": values,
                "device_ids": possible_device_ids[:num_points] # Ensure correct length
            }
        return data

class TopologyService(BaseServicePlaceholder):
    def __init__(self):
        super().__init__("TopologyService")
    # Define methods like get_current_topology as needed

class ModelRepository(BaseServicePlaceholder):
    def __init__(self):
        super().__init__("ModelRepository")
        self._models: Dict[str, Dict[str, Any]] = {}
        self._model_storage_path = Path("data/ml_models") # Define a storage path
        self._model_storage_path.mkdir(parents=True, exist_ok=True)

    async def get_all_models(self) -> Dict[str, Any]:
        logger.info("ModelRepository: Getting all models metadata.")
        # This would typically list metadata, not load all models into memory.
        # For now, returning metadata from the in-memory store.
        return { 
            model_id: { 
                "metadata": data.get("metadata"), 
                "performance": data.get("performance"),
                "status": data.get("status")
            } 
            for model_id, data in self._models.items()
        }

    async def load_model(self, model_id: str) -> Any:
        logger.info(f"ModelRepository: Loading model {model_id}.")
        model_data = self._models.get(model_id)
        if not model_data:
            logger.error(f"Model {model_id} not found in repository metadata.")
            return None
        
        # In a real scenario, this would deserialize a model object from a file
        # For this example, we simulate by checking if a placeholder path exists in metadata
        # and returning the model instance if it was stored directly during save_model (for simplicity here)
        # A more robust implementation would use joblib, pickle, or framework-specific save/load.
        model_file_path = self._model_storage_path / f"{model_id}.joblib" # Example path
        if model_file_path.exists():
            try:
                # import joblib
                loaded_model = joblib.load(model_file_path)
                logger.info(f"Successfully loaded model {model_id} from {model_file_path}")
                return loaded_model
                # logger.info(f"Simulating load of model {model_id} from {model_file_path}. Returning stored instance.")
                # return model_data.get("model_instance_placeholder") # Placeholder load
            except Exception as e:
                logger.error(f"Error loading model {model_id} from {model_file_path}: {e}")
                return None
        else:
            logger.warning(f"Model file {model_file_path} not found. Returning stored instance if available.")
            return model_data.get("model_instance_placeholder", "dummy_model_object_not_saved_to_disk")

    async def save_model(self, model_id: str, model: Any, metadata: Dict[str, Any], performance: Dict[str, Any]):
        logger.info(f"ModelRepository: Saving model {model_id}.")
        # In a real scenario, this would serialize the model and save to a file.
        # For this example, we store the model instance directly and simulate saving.
        # A more robust implementation would use joblib, pickle, or framework-specific save/load.
        model_file_path = self._model_storage_path / f"{model_id}.joblib" # Example path
        try:
            # import joblib
            joblib.dump(model, model_file_path)
            logger.info(f"Successfully saved model {model_id} to {model_file_path}")
            # pass # Simulate save
        except Exception as e:
            logger.error(f"Error saving model {model_id} to {model_file_path}: {e}")
            # Decide if this is a fatal error for the operation

        self._models[model_id] = {
            "model_instance_placeholder": model, # Store the model object itself for this placeholder
            "metadata": metadata,
            "performance": performance,
            "status": "trained",
            "model_file_path": str(model_file_path) # Store path in metadata
        }
        logger.info(f"Model {model_id} metadata stored. Simulated save to {model_file_path}.")
        return {"status": "success", "model_id": model_id, "path": str(model_file_path)}

# --- End of Placeholder Definitions ---

class NetworkMLAnalytics:
    """Advanced machine learning analytics for optical networks"""
    
    def __init__(self, telemetry_service: TelemetryService, metric_store: MetricStore, 
                 topology_service: TopologyService, model_repository: ModelRepository):
        self.telemetry_service = telemetry_service
        self.metric_store = metric_store
        self.topology_service = topology_service
        self.model_repository = model_repository
        self.trained_models = {}
        self.current_predictions = {}
    
    async def initialize(self):
        """Initialize the ML analytics engine"""
        # Load pre-trained models
        pretrained_models = await self.model_repository.get_all_models()
        
        for model_id, model_data in pretrained_models.items():
            if model_data["status"] == "trained":
                self.trained_models[model_id] = {
                    "model": await self.model_repository.load_model(model_id),
                    "metadata": model_data["metadata"],
                    "performance": model_data["performance"]
                }
        
        return {
            "status": "initialized",
            "models_loaded": len(self.trained_models)
        }
    
    async def train_anomaly_detection_model(self, config):
        """Train a new anomaly detection model"""
        # Validate configuration
        if not self._validate_model_config(config, "anomaly_detection"):
            return {
                "status": "error",
                "message": "Invalid model configuration"
            }
        
        # Get historical telemetry data
        history_days = config.get("history_days", 30)
        telemetry_data = await self.metric_store.get_metrics(
            metrics=config["metrics"],
            start_time=datetime.datetime.now() - datetime.timedelta(days=history_days),
            end_time=datetime.datetime.now(),
            resolution=config.get("resolution", "5m")
        )
        
        # Preprocess data
        processed_data = await self._preprocess_telemetry_data(telemetry_data, config)
        
        # Split into training and validation sets
        train_data, val_data = self._split_dataset(processed_data, config.get("val_split", 0.2))
        
        # Create and train model
        model_type = config["model_type"]
        
        if model_type == "isolation_forest":
            model = IsolationForest(
                n_estimators=config.get("n_estimators", 100),
                contamination=config.get("contamination", 0.01),
                random_state=42
            )
            model.fit(train_data["features"])
            
        elif model_type == "one_class_svm":
            model = OneClassSVM(
                nu=config.get("nu", 0.01),
                kernel=config.get("kernel", "rbf"),
                gamma=config.get("gamma", "auto")
            )
            model.fit(train_data["features"])
            
        elif model_type == "lstm_autoencoder":
            # Build LSTM autoencoder using PyTorch
            model = self._build_lstm_autoencoder(
                input_dim=train_data["features"].shape[1],
                encoding_dim=config.get("encoding_dim", 32),
                lstm_layers=config.get("lstm_layers", 2)
            )
            
            # Train the model
            await self._train_autoencoder(
                model=model,
                train_data=train_data,
                val_data=val_data,
                config=config
            )
        
        # Evaluate model
        performance = await self._evaluate_anomaly_model(model, val_data, config)
        
        # Save model
        model_id = str(uuid.uuid4())
        
        await self.model_repository.save_model(
            model_id=model_id,
            model=model,
            metadata={
                "name": config.get("name", f"anomaly_detection_{model_id[:8]}"),
                "description": config.get("description", "Anomaly detection model"),
                "model_type": model_type,
                "metrics": config["metrics"],
                "creation_date": datetime.datetime.now().isoformat(),
                "config": config
            },
            performance=performance
        )
        
        # Store in trained models
        self.trained_models[model_id] = {
            "model": model,
            "metadata": {
                "name": config.get("name", f"anomaly_detection_{model_id[:8]}"),
                "description": config.get("description", "Anomaly detection model"),
                "model_type": model_type,
                "metrics": config["metrics"],
                "creation_date": datetime.datetime.now().isoformat(),
                "config": config
            },
            "performance": performance
        }
        
        return {
            "status": "success",
            "model_id": model_id,
            "performance": performance
        }
    
    async def detect_anomalies(self, model_id, time_window=None):
        """Detect anomalies in current network telemetry using trained model"""
        if model_id not in self.trained_models:
            return {
                "status": "error",
                "message": f"Model not found: {model_id}"
            }
        
        model_info = self.trained_models[model_id]
        model = model_info["model"]
        metrics = model_info["metadata"]["metrics"]
        
        # Get current telemetry data (or specified time window)
        if time_window:
            start_time, end_time = time_window
        else:
            end_time = datetime.datetime.now()
            start_time = end_time - datetime.timedelta(hours=1)  # Last hour by default
        
        telemetry_data = await self.metric_store.get_metrics(
            metrics=metrics,
            start_time=start_time,
            end_time=end_time,
            resolution="1m"  # 1-minute resolution for anomaly detection
        )
        
        # Preprocess data
        processed_data = await self._preprocess_telemetry_data(
            telemetry_data, 
            model_info["metadata"]["config"]
        )
        
        # Detect anomalies
        if model_info["metadata"]["model_type"] in ["isolation_forest", "one_class_svm"]:
            # For traditional ML models
            anomaly_scores = model.decision_function(processed_data["features"])
            predictions = model.predict(processed_data["features"])
            
            # Convert scikit-learn predictions (-1 for anomalies, 1 for normal) to boolean
            anomalies = predictions == -1
            
        elif model_info["metadata"]["model_type"] == "lstm_autoencoder":
            # For autoencoder models
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            model.eval()
            
            with torch.no_grad():
                val_tensor = torch.tensor(processed_data["features"], dtype=torch.float32).to(device)
                reconstructions = model(val_tensor).cpu().numpy()
            
            # Calculate reconstruction error
            mse = np.mean(np.square(processed_data["features"] - reconstructions), axis=1)
            
            # Determine threshold for anomaly (e.g., 3 standard deviations)
            threshold = np.mean(mse) + 3 * np.std(mse)
            anomalies = mse > threshold
        
        # Prepare results
        anomaly_results = []
        for i in range(len(anomalies)):
            if anomalies[i]:
                timestamp = processed_data["timestamps"][i]
                
                # Get metrics at this timestamp
                metric_values = {
                    metric: processed_data["raw_data"][metric][i] 
                    for metric in metrics
                }
                
                current_score = None
                if model_info["metadata"]["model_type"] in ["isolation_forest", "one_class_svm"]:
                    if 'anomaly_scores' in locals() and i < len(anomaly_scores):
                        current_score = float(anomaly_scores[i])
                elif model_info["metadata"]["model_type"] == "lstm_autoencoder":
                    if 'mse' in locals() and i < len(mse):
                        current_score = float(mse[i])
                
                anomaly_results.append({
                    "timestamp": timestamp.isoformat(),
                    "score": current_score,
                    "metrics": metric_values,
                    "device_id": processed_data["device_ids"][i] if "device_ids" in processed_data and i < len(processed_data["device_ids"]) else None
                })
        
        # Store current predictions
        self.current_predictions[model_id] = {
            "timestamp": datetime.datetime.now().isoformat(),
            "time_window": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "anomaly_count": len(anomaly_results),
            "anomalies": anomaly_results
        }
        
        return {
            "status": "success",
            "model_id": model_id,
            "time_window": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "anomaly_count": len(anomaly_results),
            "anomalies": anomaly_results
        }
    
    async def train_traffic_prediction_model(self, config):
        """Train a model to predict future network traffic patterns"""
        # Validate configuration
        if not self._validate_model_config(config, "traffic_prediction"):
            return {
                "status": "error",
                "message": "Invalid model configuration"
            }
        
        # Get historical traffic data
        history_days = config.get("history_days", 30)
        traffic_data = await self.metric_store.get_metrics(
            metrics=config["metrics"],
            start_time=datetime.datetime.now() - datetime.timedelta(days=history_days),
            end_time=datetime.datetime.now(),
            resolution=config.get("resolution", "5m")
        )
        
        # Preprocess data
        processed_data = await self._preprocess_telemetry_data(traffic_data, config)
        
        # Create time series features (hour of day, day of week, etc.)
        time_features = self._create_time_features(processed_data["timestamps"])
        features = np.hstack([processed_data["features"], time_features])
        
        # Create sequences for time series prediction
        X, y = self._create_sequences(
            features, 
            processed_data["target"],
            seq_length=config.get("sequence_length", 12)  # Default: use last 12 points to predict next
        )
        
        # Split into training and validation sets
        train_size = int(len(X) * (1 - config.get("val_split", 0.2)))
        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:], y[train_size:]
        
        # Create and train model
        model_type = config["model_type"]
        
        if model_type == "lstm":
            # Build LSTM model
            model = self._build_lstm_model(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                output_size=y_train.shape[1],
                lstm_units=config.get("lstm_units", 64),
                dropout=config.get("dropout", 0.2)
            )
            
            # Train model
            await self._train_lstm_model(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                config=config
            )
            
        elif model_type == "prophet":
            if Prophet is None:
                logger.error("Prophet is not installed. Cannot train Prophet model.")
                return {
                    "status": "error",
                    "message": "Prophet is not installed. Please install prophet to use this model type."
                }
            # For Facebook Prophet
            prophet_data = self._prepare_prophet_data(
                timestamps=processed_data["timestamps"],
                values=processed_data["target"]
            )
            
            model = Prophet(
                changepoint_prior_scale=config.get("changepoint_prior_scale", 0.05),
                seasonality_mode=config.get("seasonality_mode", "multiplicative")
            )
            
            # Add relevant seasonalities
            if config.get("add_weekly_seasonality", True):
                model.add_seasonality(
                    name='weekly', 
                    period=7, 
                    fourier_order=config.get("weekly_fourier_order", 3)
                )
                
            if config.get("add_daily_seasonality", True):
                model.add_seasonality(
                    name='daily', 
                    period=1, 
                    fourier_order=config.get("daily_fourier_order", 5)
                )
            
            # Fit the model
            model.fit(prophet_data)
            
        elif model_type == "xgboost":
            if XGBRegressor is None:
                logger.error("XGBoost is not installed. Cannot train XGBoost model.")
                return {
                    "status": "error",
                    "message": "XGBoost is not installed. Please install xgboost to use this model type."
                }
            
            model = XGBRegressor(
                n_estimators=config.get("n_estimators", 100),
                learning_rate=config.get("learning_rate", 0.1),
                max_depth=config.get("max_depth", 6),
                min_child_weight=config.get("min_child_weight", 1),
                subsample=config.get("subsample", 0.8),
                colsample_bytree=config.get("colsample_bytree", 0.8),
                gamma=config.get("gamma", 0),
                objective='reg:squarederror',
                random_state=42
            )
            
            # Flatten sequences for XGBoost
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_val_flat = X_val.reshape(X_val.shape[0], -1)
            
            # Train model
            model.fit(
                X_train_flat, 
                y_train,
                eval_set=[(X_val_flat, y_val)],
                early_stopping_rounds=config.get("early_stopping_rounds", 10),
                verbose=False
            )
        
        # Evaluate model
        performance = await self._evaluate_prediction_model(
            model=model,
            model_type=model_type,
            X_val=X_val,
            y_val=y_val,
            config=config
        )
        
        # Save model
        model_id = str(uuid.uuid4())
        
        await self.model_repository.save_model(
            model_id=model_id,
            model=model,
            metadata={
                "name": config.get("name", f"traffic_prediction_{model_id[:8]}"),
                "description": config.get("description", "Traffic prediction model"),
                "model_type": model_type,
                "metrics": config["metrics"],
                "creation_date": datetime.datetime.now().isoformat(),
                "config": config
            },
            performance=performance
        )
        
        # Store in trained models
        self.trained_models[model_id] = {
            "model": model,
            "metadata": {
                "name": config.get("name", f"traffic_prediction_{model_id[:8]}"),
                "description": config.get("description", "Traffic prediction model"),
                "model_type": model_type,
                "metrics": config["metrics"],
                "creation_date": datetime.datetime.now().isoformat(),
                "config": config
            },
            "performance": performance
        }
        
        return {
            "status": "success",
            "model_id": model_id,
            "performance": performance
        }
    
    async def predict_traffic(self, model_id, horizon=24, data_window=None):
        """Predict future network traffic using a trained model"""
        if model_id not in self.trained_models:
            return {
                "status": "error",
                "message": f"Model not found: {model_id}"
            }
        
        model_info = self.trained_models[model_id]
        model = model_info["model"]
        metrics = model_info["metadata"]["metrics"]
        model_type = model_info["metadata"]["model_type"]
        
        # Get current traffic data (or specified time window)
        if data_window:
            start_time, end_time = data_window
        else:
            end_time = datetime.datetime.now()
            sequence_length = model_info["metadata"]["config"].get("sequence_length", 12)
            start_time = end_time - datetime.timedelta(hours=sequence_length)
        
        traffic_data = await self.metric_store.get_metrics(
            metrics=metrics,
            start_time=start_time,
            end_time=end_time,
            resolution="1h"  # 1-hour resolution for traffic prediction
        )
        
        # Preprocess data
        processed_data = await self._preprocess_telemetry_data(
            traffic_data, 
            model_info["metadata"]["config"]
        )
        
        # Create time series features
        time_features = self._create_time_features(processed_data["timestamps"])
        features = np.hstack([processed_data["features"], time_features])
        
        # Predict future traffic
        if model_type == "lstm":
            # Create sequence for LSTM
            X = features[-model_info["metadata"]["config"].get("sequence_length", 12):]
            X = X.reshape(1, X.shape[0], X.shape[1])
            
            # Generate predictions for each step in horizon
            predictions = []
            current_sequence = X.copy()
            
            for _ in range(horizon):
                # Predict next step
                next_pred = model(torch.tensor(current_sequence, dtype=torch.float32)).detach().numpy()[0]
                predictions.append(next_pred)
                
                # Update sequence for next prediction
                # Create time features for next timestamp
                next_timestamp = processed_data["timestamps"][-1] + datetime.timedelta(hours=1)
                next_time_features = self._create_time_features([next_timestamp])[0]
                
                # Combine prediction with time features
                next_features = np.concatenate([next_pred, next_time_features])
                
                # Update sequence by removing oldest and adding newest
                current_sequence = np.roll(current_sequence, -1, axis=1)
                current_sequence[0, -1, :] = next_features
            
            # Combine predictions
            predictions = np.array(predictions)
            
        elif model_type == "prophet":
            # For Prophet, we create a future dataframe
            future = model.make_future_dataframe(periods=horizon, freq='H')
            forecast = model.predict(future)
            
            # Extract predictions for the horizon
            predictions = forecast.iloc[-horizon:]['yhat'].values
            
        elif model_type == "xgboost":
            # For XGBoost, we need to create a sequence and incrementally predict
            sequence = features[-model_info["metadata"]["config"].get("sequence_length", 12):].flatten().reshape(1, -1)
            
            predictions = []
            for _ in range(horizon):
                # Predict next value
                next_pred = model.predict(sequence)[0]
                predictions.append(next_pred)
                
                # Update sequence for next prediction
                next_timestamp = processed_data["timestamps"][-1] + datetime.timedelta(hours=1)
                next_time_features = self._create_time_features([next_timestamp])[0]
                
                # Roll sequence and update
                sequence = np.roll(sequence, -len(next_time_features)-1)
                sequence[0, -(len(next_time_features)+1):] = np.append(next_pred, next_time_features)
            
            predictions = np.array(predictions)
        
        # Generate timestamps for predictions
        prediction_timestamps = [
            end_time + datetime.timedelta(hours=i+1)
            for i in range(horizon)
        ]
        
        # Format results
        prediction_results = []
        for i, timestamp in enumerate(prediction_timestamps):
            prediction_results.append({
                "timestamp": timestamp.isoformat(),
                "prediction": predictions[i].tolist() if isinstance(predictions[i], np.ndarray) else float(predictions[i])
            })
        
        # Store current predictions
        self.current_predictions[model_id] = {
            "timestamp": datetime.datetime.now().isoformat(),
            "prediction_horizon": horizon,
            "predictions": prediction_results
        }
        
        return {
            "status": "success",
            "model_id": model_id,
            "prediction_horizon": horizon,
            "predictions": prediction_results
        }
    
    async def analyze_network_impact(self, model_id, scenario):
        """Analyze the impact of a network scenario on predicted traffic"""
        if model_id not in self.trained_models:
            return {
                "status": "error",
                "message": f"Model not found: {model_id}"
            }
        
        # Get base prediction
        base_prediction = await self.predict_traffic(model_id)
        
        # Apply scenario modifications
        scenario_type = scenario.get("type", "capacity_change")
        impact_factor = scenario.get("impact_factor", 1.0)
        
        scenario_predictions = []
        for pred in base_prediction["predictions"]:
            modified_pred = pred.copy()
        
            if scenario_type == "capacity_change":
                # Simulate capacity change by scaling predictions
                if isinstance(modified_pred["prediction"], list):
                    modified_pred["prediction"] = [p * impact_factor for p in modified_pred["prediction"]]
                else:
                    modified_pred["prediction"] = modified_pred["prediction"] * impact_factor
            
            elif scenario_type == "failure_event":
                # Simulate failure by sharp drop followed by recovery
                if isinstance(modified_pred["prediction"], list):
                    drop_factor = max(0.1, 1.0 - impact_factor)
                    recovery_rate = scenario.get("recovery_rate", 0.2)
                
                    time_idx = base_prediction["predictions"].index(pred)
                    recovery_factor = min(1.0, drop_factor + (time_idx * recovery_rate))
                
                    modified_pred["prediction"] = [p * recovery_factor for p in modified_pred["prediction"]]
                else:
                    drop_factor = max(0.1, 1.0 - impact_factor)
                    recovery_rate = scenario.get("recovery_rate", 0.2)
                
                    time_idx = base_prediction["predictions"].index(pred)
                    recovery_factor = min(1.0, drop_factor + (time_idx * recovery_rate))
                
                    modified_pred["prediction"] = modified_pred["prediction"] * recovery_factor
        
            scenario_predictions.append(modified_pred)
        
        return {
            "status": "success",
            "model_id": model_id,
            "scenario": scenario,
            "base_predictions": base_prediction["predictions"],
            "scenario_predictions": scenario_predictions,
            "impact_analysis": {
                "average_change_percent": round(
                    (sum(s["prediction"] - b["prediction"] 
                        if not isinstance(s["prediction"], list) else 
                        sum(s["prediction"]) - sum(b["prediction"])
                        for s, b in zip(scenario_predictions, base_prediction["predictions"])) / 
                    (sum(b["prediction"] 
                        if not isinstance(b["prediction"], list) else 
                        sum(b["prediction"])
                        for b in base_prediction["predictions"]))) * 100,
                    2
                ),
                "max_impact_time": max(
                    scenario_predictions, 
                    key=lambda x: abs(x["prediction"] - 
                        base_prediction["predictions"][scenario_predictions.index(x)]["prediction"])
                        if not isinstance(x["prediction"], list) else
                        abs(sum(x["prediction"]) - 
                            sum(base_prediction["predictions"][scenario_predictions.index(x)]["prediction"]))
                )["timestamp"]
            }
        }
    
    # ---- Helper Methods ----
    
    def _validate_model_config(self, config, model_type):
        """Validate model configuration"""
        if model_type == "anomaly_detection":
            required_fields = ["metrics", "model_type"]
            valid_model_types = ["isolation_forest", "one_class_svm", "lstm_autoencoder"]
        
        elif model_type == "traffic_prediction":
            required_fields = ["metrics", "model_type"]
            valid_model_types = ["lstm", "prophet", "xgboost"]
        
        else:
            logger.error(f"Unknown model type: {model_type}")
            return False
    
        # Check required fields
        for field in required_fields:
            if field not in config:
                logger.error(f"Missing required field in configuration: {field}")
                return False
    
        # Check valid model type
        if config["model_type"] not in valid_model_types:
            logger.error(f"Invalid model type: {config['model_type']}. Must be one of: {valid_model_types}")
            return False
    
        return True

    async def _preprocess_telemetry_data(self, telemetry_data, config):
        """Preprocess telemetry data for ML models"""
        # Extract timestamps and feature values
        timestamps = telemetry_data.get("timestamps", [])
        metrics = config["metrics"]
    
        # Handle empty data
        if not timestamps or not metrics:
            logger.error("No data or metrics available for preprocessing")
            return {
                "timestamps": [],
                "features": np.array([]),
                "target": np.array([]),
                "device_ids": []
            }
    
        # Initialize numpy arrays for features and target
        n_samples = len(timestamps)
        n_features = len(metrics) - 1 if "target_metric" in config else len(metrics)
    
        features = np.zeros((n_samples, n_features))
        target = np.zeros(n_samples) if "target_metric" in config else None
        device_ids = []
    
        # Fill feature and target arrays
        for i, metric_name in enumerate(metrics):
            # Skip target metric when filling features
            if "target_metric" in config and metric_name == config["target_metric"]:
                metric_data = telemetry_data.get(metric_name, {})
                if "values" in metric_data:
                    target = np.array(metric_data["values"])
                if "device_ids" in metric_data:
                    device_ids = metric_data["device_ids"]
                continue
        
            feature_idx = i if "target_metric" not in config or i < metrics.index(config["target_metric"]) else i - 1
        
            metric_data = telemetry_data.get(metric_name, {})
            if "values" in metric_data:
                features[:, feature_idx] = np.array(metric_data["values"])
        
            # Store device IDs if available and not already set
            if not device_ids and "device_ids" in metric_data:
                device_ids = metric_data["device_ids"]
    
        # Apply feature normalization if specified
        if config.get("normalize_features", True):
            # Apply min-max scaling to [0, 1]
            for i in range(features.shape[1]):
                col_min = np.min(features[:, i])
                col_max = np.max(features[:, i])
                if col_max > col_min:  # Avoid division by zero
                    features[:, i] = (features[:, i] - col_min) / (col_max - col_min)
    
        # Return processed data with raw data for reference
        return {
            "timestamps": timestamps,
            "features": features,
            "target": target if "target_metric" in config else None,
            "device_ids": device_ids,
            "raw_data": {metric: telemetry_data.get(metric, {}).get("values", []) for metric in metrics}
        }

    def _split_dataset(self, data, val_split=0.2):
        """Split dataset into training and validation sets"""
        n_samples = len(data["timestamps"])
        split_idx = int(n_samples * (1 - val_split))
    
        train_data = {
            "timestamps": data["timestamps"][:split_idx],
            "features": data["features"][:split_idx],
            "device_ids": data["device_ids"][:split_idx] if "device_ids" in data else None
        }
    
        val_data = {
            "timestamps": data["timestamps"][split_idx:],
            "features": data["features"][split_idx:],
            "device_ids": data["device_ids"][split_idx:] if "device_ids" in data else None
        }
    
        if data["target"] is not None:
            train_data["target"] = data["target"][:split_idx]
            val_data["target"] = data["target"][split_idx:]
    
        return train_data, val_data

    def _build_lstm_autoencoder(self, input_dim, encoding_dim=32, lstm_layers=2):
        """Build LSTM autoencoder model"""
        class LSTMAutoencoder(nn.Module):
            def __init__(self, input_dim, encoding_dim, lstm_layers):
                super(LSTMAutoencoder, self).__init__()
            
                self.input_dim = input_dim
                self.encoding_dim = encoding_dim
                self.lstm_layers = lstm_layers
            
                # Encoder
                self.encoder_lstm = nn.LSTM(
                    input_size=input_dim,
                    hidden_size=encoding_dim,
                    num_layers=lstm_layers,
                    batch_first=True
                )
            
                # Decoder
                self.decoder_lstm = nn.LSTM(
                    input_size=encoding_dim,
                    hidden_size=input_dim,
                    num_layers=lstm_layers,
                    batch_first=True
                )
            
                self.output_layer = nn.Linear(input_dim, input_dim)
        
            def forward(self, x):
                # Encode
                _, (hidden, _) = self.encoder_lstm(x)
            
                # Use last hidden state
                hidden_repeated = hidden[-1].unsqueeze(1).repeat(1, x.size(1), 1)
            
                # Decode
                output, _ = self.decoder_lstm(hidden_repeated)
            
                # Final output
                reconstructed = self.output_layer(output)
            
                return reconstructed
    
        return LSTMAutoencoder(input_dim, encoding_dim, lstm_layers)

    async def _train_autoencoder(self, model, train_data, val_data, config):
        """Train an autoencoder model"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
    
        # Prepare data loaders
        train_tensor = torch.tensor(train_data["features"], dtype=torch.float32)
        train_dataset = TensorDataset(train_tensor, train_tensor)  # Input = Output for autoencoder
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.get("batch_size", 32),
            shuffle=True
        )
    
        val_tensor = torch.tensor(val_data["features"], dtype=torch.float32)
        val_dataset = TensorDataset(val_tensor, val_tensor)
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.get("batch_size", 32)
        )
    
        # Setup optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config.get("learning_rate", 0.001)
        )
    
        # Training loop
        epochs = config.get("epochs", 100)
        patience = config.get("patience", 10)
        best_val_loss = float('inf')
        patience_counter = 0
    
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
        
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
                # Forward pass
                reconstructed = model(batch_X)
                loss = F.mse_loss(reconstructed, batch_y)
            
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
                train_loss += loss.item()
        
            train_loss /= len(train_loader)
        
            # Validation
            model.eval()
            val_loss = 0.0
        
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                    reconstructed = model(batch_X)
                    loss = F.mse_loss(reconstructed, batch_y)
                
                    val_loss += loss.item()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
            logger.info(f"Epoch {epoch+1}/{epochs} - Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
    
        return model

    async def _evaluate_anomaly_model(self, model, val_data, config):
        """Evaluate an anomaly detection model"""
        model_type = config["model_type"]
    
        if model_type in ["isolation_forest", "one_class_svm"]:
            # For traditional ML models
            predictions = model.predict(val_data["features"])
            anomaly_scores = model.decision_function(val_data["features"])
        
            # Simulate some ground truth for performance metrics (normally would use labeled data)
            # For demonstration, we'll consider the most extreme scores as true anomalies (top 1%)
            threshold = np.percentile(anomaly_scores, 1)
            ground_truth = anomaly_scores <= threshold
        
            # Convert scikit-learn predictions (-1 for anomalies, 1 for normal) to boolean (True for anomaly)
            # Predictions here are boolean: True if anomaly, False if normal.
            # Scikit-learn: -1 for anomaly (True), 1 for normal (False)
            predicted_anomalies = anomaly_scores <= threshold
        
        elif model_type == "lstm_autoencoder":
            # For autoencoder
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            model.eval()
        
            with torch.no_grad():
                val_tensor = torch.tensor(val_data["features"], dtype=torch.float32).to(device)
                reconstructions = model(val_tensor).cpu().numpy()
        
            # Calculate reconstruction error
            mse = np.mean(np.square(val_data["features"] - reconstructions), axis=1)
        
            # For evaluation metrics, simulate ground truth similar to other models (e.g., top 1% of errors)
            # This 'ground_truth' is for evaluation purposes only.
            eval_threshold = np.percentile(mse, 99) # Higher MSE scores are more anomalous
            ground_truth = mse >= eval_threshold
            
            # The 'anomalies' variable calculated earlier (mse > (mean + 3*std)) can be used as predicted_anomalies
            # Or we can define predicted_anomalies based on a consistent percentile for comparison.
            # Let's use the 3-sigma rule for predicted anomalies as it's what the detection logic might use.
            dynamic_threshold = np.mean(mse) + 3 * np.std(mse)
            predicted_anomalies = mse > dynamic_threshold

        else: # Should not happen due to prior validation
            logger.error(f"Unsupported model type {model_type} in _evaluate_anomaly_model")
            return {
                "precision": 0.0, "recall": 0.0, "f1_score": 0.0, 
                "notes": "Unsupported model type"
            }

        # Calculate common metrics
        # Ensure ground_truth and predicted_anomalies are available and are boolean arrays
        if 'ground_truth' not in locals() or 'predicted_anomalies' not in locals():
            logger.error("Ground truth or predictions not generated for metric calculation.")
            return {"error": "Failed to generate ground truth or predictions."}

        if len(ground_truth) == 0 or len(predicted_anomalies) == 0:
            logger.warning("Empty ground truth or predictions array in _evaluate_anomaly_model. Returning zero metrics.")
            return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "notes": "Empty data for evaluation"}

        precision = precision_score(ground_truth, predicted_anomalies, zero_division=0)
        recall = recall_score(ground_truth, predicted_anomalies, zero_division=0)
        f1 = f1_score(ground_truth, predicted_anomalies, zero_division=0)
        
        logger.info(f"Model {model_type} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            # Add other relevant metrics like AUC if scores/probabilities are suitable
        }

    # ---- Helper Methods ----