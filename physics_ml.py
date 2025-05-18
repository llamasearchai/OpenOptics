import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, List

class OpticalPhysicsNN(nn.Module):
    """Neural network incorporating optical physics equations for accurate modeling"""
    
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 64)
        self.linear2 = nn.Linear(64, 64)
        self.output = nn.Linear(64, 4)  # Power, OSNR, Dispersion, BER
        
        # Physical constants
        self.c = 3e8  # Speed of light, m/s
        self.h = 6.63e-34  # Planck's constant, J⋅s
        
        # Configure physics constraints
        self.physics_constraints = {
            "enable_power_conservation": True,
            "enable_dispersion_constraints": True,
            "enable_osnr_ber_relationship": True,
            "enable_nonlinear_effects": True
        }
        
        # Default feature mapping
        self.feature_mapping = {
            "input_power_idx": 0,
            "distance_idx": 1,
            "wavelength_idx": 2,
            "dispersion_coef_idx": 3,
            "attenuation_idx": 4,
            "nonlinear_coef_idx": 5
        }
        
        # Default output mapping
        self.output_mapping = {
            "power_idx": 0,
            "osnr_idx": 1,
            "dispersion_idx": 2,
            "ber_idx": 3
        }
    
    def forward(self, x):
        """Forward pass with physics-based constraints"""
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        predictions = self.output(x)
        
        # Apply physics-based corrections
        predictions = self.apply_physics_constraints(predictions, x)
        return predictions
    
    def apply_physics_constraints(self, predictions, inputs):
        """Apply optical physics equations to ensure realistic values"""
        # Assuming input features are structured as:
        # [input_power_dbm, distance_km, wavelength_nm, dispersion_coef, attenuation_db_km, ...]
        
        # Create a copy for modification
        physics_corrected = predictions.clone()
        
        # Apply power conservation constraint (P_out = P_in - attenuation * distance)
        input_power = inputs[:, 0]  # Assuming first feature is input power in dBm
        distance = inputs[:, 1]     # Assuming second feature is distance in km
        attenuation = inputs[:, 4]  # Assuming fifth feature is attenuation in dB/km
        
        expected_output_power = input_power - attenuation * distance
        # Blend neural network prediction with physics formula (70% physics, 30% neural network)
        physics_corrected[:, 0] = 0.7 * expected_output_power + 0.3 * physics_corrected[:, 0]
        
        # Apply dispersion constraint (D_total = dispersion_coef * distance)
        dispersion_coef = inputs[:, 3]  # Assuming fourth feature is dispersion coefficient
        expected_dispersion = dispersion_coef * distance
        # Blend neural network prediction with physics formula
        physics_corrected[:, 2] = 0.8 * expected_dispersion + 0.2 * physics_corrected[:, 2]
        
        # Apply OSNR-BER relationship (simplified approximation)
        osnr_db = physics_corrected[:, 1]  # Second output is OSNR
        # Convert OSNR from dB to linear
        osnr_linear = 10 ** (osnr_db / 10)
        # Simplified BER calculation based on OSNR (approximation of erfc function)
        expected_ber = 0.5 * torch.exp(-osnr_linear / 2)
        # Enforce realistic BER range
        expected_ber = torch.clamp(expected_ber, min=1e-15, max=0.5)
        # Blend NN prediction with physics formula
        physics_corrected[:, 3] = 0.9 * expected_ber + 0.1 * physics_corrected[:, 3]
        
        return physics_corrected
    
    def configure_feature_mapping(self, mapping: Dict[str, int]):
        """Configure which input features correspond to which physical parameters
        
        Args:
            mapping: Dictionary mapping feature names to indices
        """
        self.feature_mapping.update(mapping)
    
    def configure_output_mapping(self, mapping: Dict[str, int]):
        """Configure which output indices correspond to which physical parameters
        
        Args:
            mapping: Dictionary mapping output names to indices
        """
        self.output_mapping.update(mapping)
    
    def enable_physics_constraint(self, constraint_name: str, enable: bool = True):
        """Enable or disable a specific physics constraint
        
        Args:
            constraint_name: Name of the constraint to configure
            enable: Whether to enable the constraint
        """
        if constraint_name in self.physics_constraints:
            self.physics_constraints[constraint_name] = enable
        else:
            raise ValueError(f"Unknown constraint: {constraint_name}")
    
    def predict_with_confidence(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make predictions with confidence estimates
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (predictions, confidence)
        """
        # Make predictions (which already include some physics blending)
        predictions = self.forward(x)
        
        # Initialize confidences to a default high value (e.g., 1.0)
        confidences = torch.ones_like(predictions)

        # --- Power Output Confidence ---
        if self.physics_constraints.get("enable_power_conservation", False):
            input_power_dbm = x[:, self.feature_mapping["input_power_idx"]] 
            distance_km = x[:, self.feature_mapping["distance_idx"]]
            attenuation_db_km = x[:, self.feature_mapping["attenuation_idx"]]
            
            # Pure physics calculation for output power
            pure_physics_output_power = input_power_dbm - attenuation_db_km * distance_km
            
            # Deviation of the NN's blended prediction from pure physics
            power_deviation = torch.abs(predictions[:, self.output_mapping["power_idx"]] - pure_physics_output_power)
            
            # Convert deviation to confidence (e.g., exp(-k * deviation^2)). 
            # Larger deviation = lower confidence. Max confidence 1.
            # The scaling factor 'k' needs tuning. For example, if a 2dB deviation means ~0.3 confidence:
            # exp(-k * 2^2) = 0.3 => -4k = ln(0.3) => k = -ln(0.3)/4 approx 0.3
            k_power = 0.3 
            confidences[:, self.output_mapping["power_idx"]] = torch.exp(-k_power * power_deviation**2)

        # --- Dispersion Confidence ---
        if self.physics_constraints.get("enable_dispersion_constraints", False):
            distance_km = x[:, self.feature_mapping["distance_idx"]]
            dispersion_coef = x[:, self.feature_mapping["dispersion_coef_idx"]]
            
            pure_physics_dispersion = dispersion_coef * distance_km
            dispersion_deviation = torch.abs(predictions[:, self.output_mapping["dispersion_idx"]] - pure_physics_dispersion)
            
            # Similar confidence logic, k_dispersion might need different tuning
            # If 100 ps deviation means ~0.3 confidence: k = -ln(0.3)/(100^2) approx 1.2e-4
            k_dispersion = 1.2e-4
            confidences[:, self.output_mapping["dispersion_idx"]] = torch.exp(-k_dispersion * dispersion_deviation**2)

        # --- BER Confidence (based on OSNR-BER relationship consistency) ---
        if self.physics_constraints.get("enable_osnr_ber_relationship", False):
            predicted_osnr_db = predictions[:, self.output_mapping["osnr_idx"]]
            predicted_ber = predictions[:, self.output_mapping["ber_idx"]]

            # Pure physics calculation for BER from predicted OSNR
            osnr_linear = 10 ** (predicted_osnr_db / 10)
            pure_physics_ber = 0.5 * torch.exp(-osnr_linear / 2) # Simplified Q-function approx.
            pure_physics_ber = torch.clamp(pure_physics_ber, min=1e-15, max=0.5)
            
            # Deviation in orders of magnitude might be more relevant for BER
            # Use a small epsilon to avoid log(0)
            ber_deviation = torch.abs(torch.log10(predicted_ber + 1e-16) - torch.log10(pure_physics_ber + 1e-16))
            
            # If 1 order of magnitude deviation means ~0.3 confidence: k = -ln(0.3)/(1^2) approx 1.2
            k_ber = 1.2
            confidences[:, self.output_mapping["ber_idx"]] = torch.exp(-k_ber * ber_deviation**2)
        
        # OSNR confidence could be harder to define without a direct physics equation for it 
        # (it's often an outcome of many factors or a target itself).
        # For now, OSNR confidence could remain 1.0 or be an average of others, or derived from input variance if model is Bayesian.
        # Let's leave OSNR confidence at default 1.0 for this example as its own physical calculation is not as direct here.

        # Clamp all confidences to be between 0 and 1
        confidences = torch.clamp(confidences, min=0.0, max=1.0)
        
        return predictions, confidences
    
    def save_model(self, path: str):
        """Save the model to disk
        
        Args:
            path: Path to save the model
        """
        model_state = {
            'model_state_dict': self.state_dict(),
            'feature_mapping': self.feature_mapping,
            'output_mapping': self.output_mapping,
            'physics_constraints': self.physics_constraints
        }
        torch.save(model_state, path)
    
    @classmethod
    def load_model(cls, path: str) -> 'OpticalPhysicsNN':
        """Load the model from disk
        
        Args:
            path: Path to load the model from
            
        Returns:
            Loaded OpticalPhysicsNN model
        """
        model_state = torch.load(path)
        
        # Create new model instance
        model = cls()
        
        # Load state dict
        model.load_state_dict(model_state['model_state_dict'])
        
        # Load configuration
        model.feature_mapping = model_state['feature_mapping']
        model.output_mapping = model_state['output_mapping']
        model.physics_constraints = model_state['physics_constraints']
        
        return model

class OpticalPhysicsEnsemble:
    """Ensemble of optical physics neural networks for improved prediction"""
    
    def __init__(self, num_models: int = 5, input_dims: int = 10, hidden_dims: int = 64, output_dims: int = 4):
        """Initialize the ensemble
        
        Args:
            num_models: Number of models in the ensemble
            input_dims: Input feature dimensions
            hidden_dims: Hidden layer dimensions
            output_dims: Output dimensions
        """
        self.models = [
            OpticalPhysicsNN() 
            for _ in range(num_models)
        ]
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through all models and combine results
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (ensemble prediction, prediction variance)
        """
        # Collect predictions from all models
        all_predictions = torch.stack([model(x) for model in self.models])
        
        # Calculate mean and variance across models
        ensemble_prediction = torch.mean(all_predictions, dim=0)
        prediction_variance = torch.var(all_predictions, dim=0)
        
        return ensemble_prediction, prediction_variance
    
    def train(self, train_loader, test_loader, num_epochs: int = 100, learning_rate: float = 0.001):
        """Train all models in the ensemble
        
        Args:
            train_loader: DataLoader for training data
            test_loader: DataLoader for testing data
            num_epochs: Number of epochs to train
            learning_rate: Learning rate for optimizer
        """
        for i, model in enumerate(self.models):
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()
            
            print(f"Training model {i+1}/{len(self.models)}...")
            
            for epoch in range(num_epochs):
                # Training loop
                model.train()
                train_loss = 0.0
                for inputs, targets in train_loader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                # Evaluation
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for inputs, targets in test_loader:
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item()
                
                # Print progress
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, "
                          f"Train Loss: {train_loss/len(train_loader):.4f}, "
                          f"Val Loss: {val_loss/len(test_loader):.4f}")
    
    def save_ensemble(self, path_prefix: str):
        """Save all models in the ensemble
        
        Args:
            path_prefix: Prefix for model paths
        """
        for i, model in enumerate(self.models):
            model.save_model(f"{path_prefix}_model_{i}.pt")
    
    @classmethod
    def load_ensemble(cls, path_prefix: str, num_models: int) -> 'OpticalPhysicsEnsemble':
        """Load ensemble from disk
        
        Args:
            path_prefix: Prefix for model paths
            num_models: Number of models to load
            
        Returns:
            Loaded ensemble
        """
        ensemble = cls(num_models=0)  # Create empty ensemble
        
        for i in range(num_models):
            model_path = f"{path_prefix}_model_{i}.pt"
            model = OpticalPhysicsNN.load_model(model_path)
            ensemble.models.append(model)
        
        return ensemble