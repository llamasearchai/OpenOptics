"""
Coherent Link Simulator Module

This module simulates coherent optical communication links, modeling effects like:
- Chromatic dispersion
- Nonlinear phase noise
- OSNR degradation
- Digital signal processing
- Various modulation formats 

The simulator provides performance metrics like SNR, BER, and constellation diagrams.
"""

import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from typing import Dict, List, Any, Tuple
import asyncio
import logging
import datetime
import random
import json
import math
from scipy.special import erfc
import uuid

logger = logging.getLogger(__name__)

class CoherentLinkSimulator:
    """Simulator for coherent optical links"""
    
    def __init__(self):
        """Initialize the coherent link simulator"""
        self.c = 3e8  # Speed of light, m/s
        self.h = 6.63e-34  # Planck's constant, J⋅s
        self.current_config: Dict[str, Any] = {}
        self.active_controllers: Dict[str, Any] = {}
        
    async def simulate(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate a coherent optical link with the given parameters
        
        Args:
            config: Simulation configuration dictionary
            
        Returns:
            Simulation results including SNR, BER, etc.
        """
        # Extract parameters from config (with defaults)
        modulation = config.get("modulation", "16qam").lower()
        symbol_rate = config.get("symbol_rate", 32e9)  # Baud
        tx_power_dbm = config.get("tx_power_dbm", 0)
        osnr = config.get("osnr", 25)  # dB
        span_length = config.get("span_length", 80)  # km
        spans = config.get("spans", 1)
        dispersion = config.get("dispersion", 16.5)  # ps/nm/km
        alpha = config.get("alpha", 0.2)  # dB/km
        gamma = config.get("gamma", 1.3)  # 1/W/km
        linewidth = config.get("linewidth", 100e3)  # Hz
        edfa_gain = config.get("edfa_gain", 16)  # dB
        edfa_nf = config.get("edfa_nf", 5)  # dB
        apply_dsp = config.get("apply_dsp", True)
        
        # Convert to SI units
        tx_power = 10**(tx_power_dbm/10) * 1e-3  # W
        
        # Get modulation parameters
        mod_params = self._get_modulation_params(modulation)
        
        # Modulation
        bits_per_symbol = mod_params["bits_per_symbol"]
        symbols = self._generate_random_symbols(1024, mod_params["constellation"])
        
        # Fiber propagation
        received_symbols, propagation_details = self._simulate_fiber_propagation(
            symbols, 
            tx_power, 
            osnr,  # This OSNR is likely an *input target* OSNR or a budget, not post-propagation
            span_length, 
            spans, 
            dispersion, 
            alpha, 
            gamma, 
            linewidth, 
            edfa_gain, 
            edfa_nf, 
            symbol_rate,
            mod_params["constellation"],
            apply_dsp
        )
        
        # Calculate performance metrics
        snr_linear = self._calculate_snr(symbols, received_symbols) # This calculates based on symbol deviation
                                                                    # Potentially use propagation_details.output_osnr_linear if available
        snr_db = 10 * np.log10(snr_linear)
        evm_percent = self._calculate_evm(received_symbols, mod_params["constellation"]) * 100
        theoretical_ber = self._calculate_theoretical_ber(snr_linear, modulation)
        bit_rate_gbps = symbol_rate * bits_per_symbol / 1e9  # Gbps
        
        # Calculate Q-factor
        q_factor = 20 * np.log10(np.sqrt(2) * erfc(2 * theoretical_ber))
        
        # Power efficiency
        power_efficiency = bit_rate_gbps / (tx_power * 1000)  # Gbps/W
        
        # Chromatic dispersion details
        total_dispersion = dispersion * span_length * spans  # ps/nm
        
        # Nonlinear phase
        nonlinear_phase = gamma * tx_power * span_length * spans  # radians
        
        # Reach estimate
        reach_estimate = self._estimate_reach(snr_db, modulation)
        
        # Generate timestamp
        timestamp = datetime.datetime.now().isoformat()
        
        # Return results
        return {
            "timestamp": timestamp,
            "modulation": modulation.upper(),
            "bits_per_symbol": bits_per_symbol,
            "symbol_rate_gbaud": symbol_rate / 1e9,
            "bit_rate_gbps": bit_rate_gbps,
            "tx_power_dbm": tx_power_dbm,
            "tx_power_mw": tx_power * 1000,
            "osnr_db": osnr,
            "spans": spans,
            "span_length_km": span_length,
            "total_length_km": span_length * spans,
            "dispersion_ps_nm_km": dispersion,
            "total_dispersion_ps_nm": total_dispersion,
            "alpha_db_km": alpha,
            "gamma_w_km": gamma,
            "nonlinear_phase_rad": nonlinear_phase,
            "linewidth_khz": linewidth / 1e3,
            "edfa_gain_db": edfa_gain,
            "edfa_nf_db": edfa_nf,
            "snr_linear": snr_linear,
            "snr_db": snr_db,
            "evm_percent": evm_percent,
            "theoretical_ber": theoretical_ber,
            "q_factor_db": q_factor,
            "power_efficiency_gbps_w": power_efficiency,
            "reach_estimate_km": reach_estimate,
            "dsp_applied": apply_dsp,
            "propagation_details": propagation_details, # Add detailed propagation results
            "symbols": symbols.tolist(),
            "received_symbols": received_symbols.tolist()
        }
    
    async def generate_constellation_plot(self, symbols: List[complex], received_symbols: List[complex], modulation: str = "QAM") -> str:
        """Generate a constellation plot and return as base64 encoded string."""
        plt.figure(figsize=(8, 8))
        
        # Plot received symbols
        received_symbols_np = np.array(received_symbols)
        plt.scatter(received_symbols_np.real, received_symbols_np.imag, c='blue', marker='.', label='Received Symbols', alpha=0.7)
        
        # Plot ideal constellation points if available (or could be passed)
        # For simplicity, this example doesn't explicitly overlay ideal points from _get_modulation_params
        # but a more complete version would.
        # symbols_np = np.array(symbols) # These are the originally generated ideal symbols, could plot them too.
        # plt.scatter(symbols_np.real, symbols_np.imag, c='red', marker='x', label='Ideal Symbols', s=100)

        plt.title(f'{modulation.upper()} Constellation Diagram')
        plt.xlabel("In-Phase")
        plt.ylabel("Quadrature")
        plt.axhline(0, color='black', lw=0.5)
        plt.axvline(0, color='black', lw=0.5)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.axis('equal') # Ensure aspect ratio is equal

        # Save plot to a BytesIO object
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close() # Close the figure to free memory

        return img_base64

    def _get_modulation_params(self, modulation: str) -> Dict[str, Any]:
        """Get modulation parameters
        
        Args:
            modulation: Modulation format (qpsk, 8qam, 16qam, 64qam, 256qam)
            
        Returns:
            Dictionary of modulation parameters
        """
        if modulation == "qpsk":
            # QPSK constellation points (normalized)
            constellation = np.array([
                1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j
            ]) / np.sqrt(2)
            bits_per_symbol = 2
            
        elif modulation == "8qam":
            # 8-QAM constellation (circular arrangement)
            angles = np.arange(0, 8) * (2 * np.pi / 8)
            constellation = np.exp(1j * angles)
            bits_per_symbol = 3
            
        elif modulation == "16qam":
            # 16-QAM constellation (square 4x4 grid)
            x = np.array([-3, -1, 1, 3])
            real, imag = np.meshgrid(x, x)
            constellation = (real.flatten() + 1j * imag.flatten()) / np.sqrt(10)
            bits_per_symbol = 4
            
        elif modulation == "64qam":
            # 64-QAM constellation (square 8x8 grid)
            x = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
            real, imag = np.meshgrid(x, x)
            constellation = (real.flatten() + 1j * imag.flatten()) / np.sqrt(42)
            bits_per_symbol = 6
            
        elif modulation == "256qam":
            # 256-QAM constellation (square 16x16 grid)
            x = np.arange(-15, 16, 2)
            real, imag = np.meshgrid(x, x)
            constellation = (real.flatten() + 1j * imag.flatten()) / np.sqrt(170)
            bits_per_symbol = 8
            
        else:
            raise ValueError(f"Unsupported modulation format: {modulation}")
        
        return {
            "constellation": constellation,
            "bits_per_symbol": bits_per_symbol
        }
    
    async def run_parameter_sweep(self, base_config: Dict[str, Any], sweep_parameter: str, sweep_values: List[Any]) -> Dict[str, List[float]]:
        """Run a parameter sweep and return performance metrics
        
        Args:
            base_config: Base simulation configuration dictionary
            sweep_parameter: Name of the parameter to sweep
            sweep_values: List of parameter values to iterate through
            
        Returns:
            Dictionary with parameter values and corresponding performance metrics
        """
        # Create parameter values array
        # param_values = np.arange(min_value, max_value + step/2, step) # Old way
        param_values_list = list(sweep_values) # Use the provided list
        
        # Initialize result arrays
        snr_values = []
        ber_values = []
        data_rate_values = []
        reach_values = []
        
        # Default parameters
        # default_params = self.current_config.copy() # Old way
        # if not default_params: 
        #     logger.warning("Sweeping with empty base current_config. Using minimal defaults provided by base_config or hardcoded.")
        #     default_params = base_config or { 
        #         "modulation": "16qam", "symbol_rate": 32e9, "tx_power_dbm": 0,
        #         "osnr": 25, "span_length": 80, "spans": 1, "dispersion": 16.5,
        #         "alpha": 0.2, "gamma": 1.3, "linewidth": 100e3, "edfa_gain": 16,
        #         "edfa_nf": 5, "apply_dsp": True
        #     }
        default_params = base_config # Use the provided base_config for sweep

        # Run simulation for each parameter value
        for value in param_values_list:
            # Update only the swept parameter
            params = default_params.copy()
            params[sweep_parameter] = value
            
            # Run simulation with these parameters
            result = await self.simulate(params)
            
            # Store results
            snr_values.append(result['snr_db'])
            ber_values.append(result['theoretical_ber'])
            data_rate_values.append(result['bit_rate_gbps'])
            reach_values.append(result['reach_estimate_km'])
        
        # Return results as a dictionary
        return {
            'param_values': param_values_list,
            'snr_values': snr_values,
            'ber_values': ber_values,
            'data_rate_values': data_rate_values,
            'reach_values': reach_values
        }
    
    async def create_adaptive_controller(self, controller_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create an adaptive controller for this device
        
        Args:
            controller_type: Type of controller (snr_optimizer, modulation_adaptor, power_optimizer)
            config: Controller configuration
            
        Returns:
            Controller information
        """
        # Create a unique ID for the controller
        controller_id = str(uuid.uuid4())
        
        # Controller configuration
        controller = {
            'id': controller_id,
            'type': controller_type,
            'config': config,
            'active': True,
            'created_at': datetime.datetime.now().isoformat(),
            'last_update': None
        }
        
        # Store controller
        self.active_controllers[controller_id] = controller
        
        # Start controller in the background
        if controller_type == 'snr_optimizer':
            asyncio.create_task(self._run_snr_optimizer(controller_id, config))
        elif controller_type == 'modulation_adaptor':
            asyncio.create_task(self._run_modulation_adaptor(controller_id, config))
        elif controller_type == 'power_optimizer':
            asyncio.create_task(self._run_power_optimizer(controller_id, config))
        else:
            logger.warning(f"Unknown controller type: {controller_type}")
            self.active_controllers[controller_id]['active'] = False
            return {
                'status': 'error',
                'message': f"Unknown controller type: {controller_type}"
            }
        
        return {
            'status': 'success',
            'controller': controller
        }
    
    async def stop_controller(self, controller_id: str) -> Dict[str, Any]:
        """Stop an active controller
        
        Args:
            controller_id: Controller ID
            
        Returns:
            Status information
        """
        if controller_id not in self.active_controllers:
            return {
                'status': 'error',
                'message': f"Controller not found: {controller_id}"
            }
        
        # Mark controller as inactive
        self.active_controllers[controller_id]['active'] = False
        
        return {
            'status': 'success',
            'message': f"Controller {controller_id} stopped"
        }
    
    async def _run_snr_optimizer(self, controller_id: str, config: Dict[str, Any]):
        """Run the SNR optimizer controller
        
        Args:
            controller_id: Controller ID
            config: Controller configuration
        """
        update_interval = config.get('update_interval', 60)  # Default 60 seconds
        
        logger.info(f"Starting SNR optimizer (ID: {controller_id}) with update interval {update_interval}s")
        
        while self.active_controllers.get(controller_id, {}).get('active', False):
            try:
                # Current tx power
                current_power = self.current_config.get('tx_power', 0)
                
                # Try different power levels around the current one
                test_powers = [
                    current_power * 0.9,
                    current_power * 0.95,
                    current_power,
                    current_power * 1.05,
                    current_power * 1.1
                ]
                
                best_power = current_power
                best_snr = 0
                
                # Test each power level
                for power in test_powers:
                    # Skip if power is outside allowed range
                    if power < 0.1 or power > 5.0:  # Example limits
                        continue
                    
                    # Run simulation with this power
                    params = self.current_config.copy()
                    params['tx_power'] = power
                    result = await self.simulate(params)
                    
                    # Check if SNR is better
                    if result['snr_db'] > best_snr:
                        best_snr = result['snr_db']
                        best_power = power
                
                # If we found a better power level, apply it
                if best_power != current_power:
                    logger.info(f"SNR optimizer changing tx_power from {current_power} to {best_power}")
                    self.current_config['tx_power'] = best_power
                
                # Update last update timestamp
                self.active_controllers[controller_id]['last_update'] = datetime.datetime.now().isoformat()
                
                # Wait for next update
                await asyncio.sleep(update_interval)
                
            except Exception as e:
                logger.error(f"Error in SNR optimizer: {e}")
                await asyncio.sleep(update_interval)
    
    async def _run_modulation_adaptor(self, controller_id: str, config: Dict[str, Any]):
        """Run the modulation adaptor controller
        
        Args:
            controller_id: Controller ID
            config: Controller configuration
        """
        update_interval = config.get('update_interval', 300)  # Default 5 minutes
        snr_margin = config.get('snr_margin', 3.0)  # Default 3 dB margin
        
        logger.info(f"Starting modulation adaptor (ID: {controller_id}) with update interval {update_interval}s")
        
        # Modulation formats with minimum required SNR
        modulation_snr_requirements = {
            'qpsk': 13.5,
            '8qam': 16.5,
            '16qam': 20.5,
            '64qam': 26.5,
            '256qam': 32.5
        }
        
        while self.active_controllers.get(controller_id, {}).get('active', False):
            try:
                # Run simulation with current parameters
                result = await self.simulate(self.current_config)
                current_snr = result['snr_db']
                current_mod = self.current_config.get('modulation', 'qpsk')
                
                # Determine best modulation format based on current SNR
                best_mod = None
                for mod, req_snr in modulation_snr_requirements.items():
                    if current_snr >= (req_snr + snr_margin):
                        best_mod = mod
                
                # If no suitable modulation found, use QPSK (most robust)
                if best_mod is None:
                    best_mod = 'qpsk'
                
                # Change modulation if different from current
                if best_mod != current_mod:
                    logger.info(f"Modulation adaptor changing modulation from {current_mod} to {best_mod}")
                    self.current_config['modulation'] = best_mod
                
                # Update last update timestamp
                self.active_controllers[controller_id]['last_update'] = datetime.datetime.now().isoformat()
                
                # Wait for next update
                await asyncio.sleep(update_interval)
                
            except Exception as e:
                logger.error(f"Error in modulation adaptor: {e}")
                await asyncio.sleep(update_interval)
    
    async def _run_power_optimizer(self, controller_id: str, config: Dict[str, Any]):
        """Run the power optimizer controller
        
        Args:
            controller_id: Controller ID
            config: Controller configuration
        """
        update_interval = config.get('update_interval', 120)  # Default 2 minutes
        min_snr = config.get('min_snr', 15.0)  # Default 15 dB minimum SNR
        
        logger.info(f"Starting power optimizer (ID: {controller_id}) with update interval {update_interval}s")
        
        while self.active_controllers.get(controller_id, {}).get('active', False):
            try:
                # Current tx power
                current_power = self.current_config.get('tx_power', 0)
                
                # Run simulation with current parameters
                result = await self.simulate(self.current_config)
                current_snr = result['snr_db']
                
                # If SNR is above minimum + margin, try to reduce power
                if current_snr > (min_snr + 2.0):
                    # Try reducing power by 5%
                    new_power = current_power * 0.95
                    
                    # Check if this is still above minimum power
                    if new_power >= 0.1:  # Example minimum power
                        # Test new power level
                        params = self.current_config.copy()
                        params['tx_power'] = new_power
                        new_result = await self.simulate(params)
                        
                        # If SNR is still above minimum, apply the new power
                        if new_result['snr_db'] >= min_snr:
                            logger.info(f"Power optimizer reducing tx_power from {current_power} to {new_power}")
                            self.current_config['tx_power'] = new_power
                
                # If SNR is below minimum, increase power
                elif current_snr < min_snr:
                    # Try increasing power by 5%
                    new_power = current_power * 1.05
                    
                    # Check if this is still below maximum power
                    if new_power <= 5.0:  # Example maximum power
                        logger.info(f"Power optimizer increasing tx_power from {current_power} to {new_power}")
                        self.current_config['tx_power'] = new_power
                
                # Update last update timestamp
                self.active_controllers[controller_id]['last_update'] = datetime.datetime.now().isoformat()
                
                # Wait for next update
                await asyncio.sleep(update_interval)
                
            except Exception as e:
                logger.error(f"Error in power optimizer: {e}")
                await asyncio.sleep(update_interval)

    def _simulate_fiber_propagation(self, symbols: np.ndarray, tx_power_w: float, 
                                  target_osnr_db: float, # Renamed for clarity
                                  span_length_km: float, spans: int, 
                                  dispersion_ps_nm_km: float, alpha_db_km: float, gamma_1_w_km: float,
                                  linewidth_hz: float, edfa_gain_db: float, edfa_nf_db: float,
                                  symbol_rate_baud: float, ideal_constellation: np.ndarray, apply_dsp: bool
                                 ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Simulate fiber propagation effects on the signal."""
        current_symbols = symbols.copy()
        current_power_w = tx_power_w
        accumulated_dispersion_ps_nm = 0
        accumulated_nonlinear_phase_rad = 0
        total_attenuation_db = 0
        
        # Constants
        h = 6.626e-34  # Planck's constant
        center_wavelength_m = 1550e-9 # Assume C-band
        freq_hz = self.c / center_wavelength_m
        bw_ref_nm = 0.1 # Reference bandwidth for OSNR in nm (12.5 GHz)

        log_details = []

        for i in range(spans):
            # --- Attenuation ---
            span_loss_db = alpha_db_km * span_length_km
            total_attenuation_db += span_loss_db
            power_after_fiber_w = current_power_w * (10**(-span_loss_db / 10))
            log_details.append(f"Span {i+1}: Power after fiber {10*np.log10(power_after_fiber_w*1000):.2f} dBm")

            # --- Chromatic Dispersion (CD) ---
            # This is typically compensated by DSP. Here we track the accumulated value.
            # The actual phase distortion on symbols is complex to model simply here without spectral simulation.
            accumulated_dispersion_ps_nm += dispersion_ps_nm_km * span_length_km
            # Simplified: Add some phase noise proportional to dispersion if DSP is off
            if not apply_dsp:
                disp_phase_error = np.random.normal(0, 0.01 * np.sqrt(abs(accumulated_dispersion_ps_nm)), len(current_symbols))
                current_symbols *= np.exp(1j * disp_phase_error)

            # --- Nonlinear Effects (Simplified Self-Phase Modulation - SPM) ---
            # Effective length for nonlinearity
            alpha_linear_per_km = (alpha_db_km / (10 * np.log10(np.exp(1)))) / 1000 # Convert dB/km to 1/m
            L_eff_km = (1 - np.exp(-alpha_linear_per_km * span_length_km * 1000)) / alpha_linear_per_km / 1000 if alpha_linear_per_km > 0 else span_length_km
            
            # Average power in the span for SPM calculation (can be simplified)
            # P_avg_w = current_power_w * (L_eff_km / span_length_km) if span_length_km > 0 else current_power_w 
            # Simplified: Use power at start of span for average effect estimation
            nonlinear_phase_shift_span_rad = gamma_1_w_km * current_power_w * L_eff_km
            accumulated_nonlinear_phase_rad += nonlinear_phase_shift_span_rad
            # Simplified: Add some phase noise proportional to SPM if DSP is off
            if not apply_dsp:
                 spm_phase_error = np.random.normal(0, 0.05 * np.sqrt(abs(nonlinear_phase_shift_span_rad)), len(current_symbols))
                 current_symbols *= np.exp(1j * spm_phase_error)
            log_details.append(f"Span {i+1}: Acc. CD {accumulated_dispersion_ps_nm:.2f} ps/nm, Acc. NL Phase {accumulated_nonlinear_phase_rad:.2f} rad")

            # --- Amplifier (EDFA) --- 
            if i < spans: # Amplifier after each span, except possibly the last if pre-amp is separate
                power_at_edfa_input_w = power_after_fiber_w
                
                # ASE noise power density (double-sided)
                nf_linear = 10**(edfa_nf_db / 10)
                gain_linear = 10**(edfa_gain_db / 10)
                # P_ase_density_w_hz = h * freq_hz * nf_linear * (gain_linear -1) # This is spectral density
                # P_ase_in_bw_ref_w = P_ase_density_w_hz * (self.c / center_wavelength_m**2) * (bw_ref_nm * 1e-9) # ASE in 0.1nm BW
                
                # Simplified ASE noise addition related to OSNR concept: recalculate OSNR
                # Signal power after gain
                signal_power_out_edfa_w = power_at_edfa_input_w * gain_linear
                
                # ASE power in reference bandwidth (0.1nm)
                # P_ase_total_watts = N_ase * h * nu * B_o, where N_ase = (G-1) * NF
                # B_o (optical bandwidth) for OSNR is often taken as 12.5 GHz (0.1 nm at 1550nm)
                optical_bandwidth_hz = (self.c / (center_wavelength_m**2)) * (bw_ref_nm * 1e-9)
                p_ase_w = (gain_linear -1) * nf_linear * h * freq_hz * optical_bandwidth_hz

                # Add Gaussian noise based on this P_ase. Variance is P_ase/2 per quadrature for complex noise.
                # Noise is added to the amplified signal.
                noise_variance_per_quad = p_ase_w / 2 
                # Scale noise relative to the amplified signal power for symbol impact
                # This step requires relating P_ase to symbol variance. 
                # A common approach is to use P_ase to calculate OSNR, then map OSNR to noise variance on symbols.
                # P_signal_out_edfa_w is the power of the *symbols*. If symbols are normalized, need to scale noise.
                # Let E_s be average symbol energy. P_signal_out_edfa_w = E_s * symbol_rate_baud.
                # Noise variance on symbols: sigma^2 = N0_total / (2 * E_s_normalized_avg_power) where N0_total is total noise power density at receiver.
                # This is becoming very complex. Let's simplify by directly adding noise scaled by a factor that reflects OSNR degradation.
                
                # Simplified noise addition: model noise to achieve a certain OSNR degradation per amp or maintain overall OSNR
                # This is tricky. Instead of target_osnr_db directly, let's use it as a reference. 
                # The edfa_nf is the primary source of OSNR degradation.
                # Noise power added by one EDFA: P_N = (G-1)*NF*h*nu*B_ref
                # The effective noise power impacting symbols also depends on receiver bandwidth (symbol_rate_baud)
                effective_noise_power_w = nf_linear * h * freq_hz * symbol_rate_baud # (G-1) is for total ASE, NF is for input referred noise for SNR calc
                
                # Add complex Gaussian noise. Variance = noise_power_w / 2 for each quadrature if symbols are normalized to unit average power.
                # Assuming current_symbols are normalized. We need to scale noise relative to current_power_w at EDFA input (signal part)
                # To maintain analytical tractability without full physical layer sim:
                # Let's assume the target_osnr_db is what we'd get if only this noise source existed over the link.
                # For a single span and amp influencing symbols directly:
                # SNR_span = P_signal_in_edfa / P_ase_effective_at_symbol_rate 
                # P_ase_effective_at_symbol_rate = NF_linear * h * nu * SymbolRate (input referred noise power)
                # SNR_linear_span = current_power_w / (NF_linear * h * freq_hz * symbol_rate_baud) 
                # noise_std_dev_per_span = 1 / np.sqrt(SNR_linear_span) if SNR_linear_span > 0 else 1.0
                
                # Simpler approximation for noise impact on normalized symbols for this placeholder:
                # Let noise_factor be related to NF. Higher NF = more noise.
                noise_factor = (nf_linear / 10.0) * 0.05 # Arbitrary scaling for placeholder
                noise_std_dev = noise_factor * np.sqrt(1/gain_linear) # Noise less impactful if signal is stronger post-gain, relative to normalized symbols.
                
                real_noise = np.random.normal(0, noise_std_dev, len(current_symbols))
                imag_noise = np.random.normal(0, noise_std_dev, len(current_symbols))
                current_symbols += (real_noise + 1j * imag_noise)

                current_power_w = signal_power_out_edfa_w # Update power for next span
                log_details.append(f"Span {i+1}: EDFA. Power after gain {10*np.log10(current_power_w*1000):.2f} dBm. Added noise (std dev ~{noise_std_dev:.3f})")
            else: # No EDFA after last span (or handled by pre-amplifier at receiver, not modeled here)
                current_power_w = power_after_fiber_w

        # --- Linewidth / Phase Noise (simplified) ---
        # Wiener process for phase noise, total variance = 2 * pi * linewidth * T_symbol * num_symbols
        # This is complex for symbol-by-symbol. Let's add cumulative phase noise at the end.
        total_symbol_duration = len(symbols) / symbol_rate_baud
        phase_noise_variance = 2 * np.pi * linewidth_hz * total_symbol_duration / len(symbols) # variance per symbol time
        # Apply as a random phase rotation to each symbol, assuming it accumulates incoherently for simplicity for now
        # More accurately, it's a random walk. Add a final random phase offset based on total variance.
        total_phase_std_dev = np.sqrt(2 * np.pi * linewidth_hz * (len(symbols) / symbol_rate_baud)) # Std dev of total phase drift
        # This is not quite right. Let's apply per-symbol phase jitter based on linewidth * Tsymbol
        per_symbol_phase_variance = 2 * np.pi * linewidth_hz * (1/symbol_rate_baud)
        phase_jitter = np.random.normal(0, np.sqrt(per_symbol_phase_variance), len(current_symbols))
        if not apply_dsp: # Assume DSP corrects common phase error, but some jitter remains
             current_symbols *= np.exp(1j * phase_jitter)
        log_details.append(f"Applied phase jitter due to linewidth {linewidth_hz/1e3}kHz (std dev ~{np.sqrt(per_symbol_phase_variance):.3f} rad per symbol)")

        # --- Receiver Noise (matches target_osnr_db if all other noise sources were zero) ---
        # This is an alternative way to add noise to meet a link-level OSNR.
        # However, we've added EDFA noise. So, target_osnr_db should be an outcome, not an input here.
        # Let's calculate the OSNR from the components.
        # Final signal power at receiver (before DSP gain)
        final_signal_power_w = current_power_w
        
        # Total accumulated ASE power (simplified by summing per-amplifier contributions assuming same ref bandwidth)
        # P_ase_total_receiver_w = spans * (10**(edfa_gain_db/10) -1) * (10**(edfa_nf_db/10)) * h * freq_hz * optical_bandwidth_hz
        # A more common way: OSNR_out = P_in / (F_sys * h * nu * B_ref) where F_sys is system noise figure.
        # For cascaded EDFAs: F_total approx = N_amps * NF_single_amp (if gain compensates loss exactly each span)
        system_nf_linear = spans * nf_linear if spans > 0 else nf_linear # Approximation
        total_ase_power_in_bw_ref_w = system_nf_linear * h * freq_hz * optical_bandwidth_hz
        # This ASE is relative to the input of the *first* amplifier if F_sys is input-referred for the cascade.
        # To use it at the receiver, it needs to be amplified by total gain, or signal power needs to be input power.
        # More robust: calculate OSNR based on signal power at receiver and total noise power at receiver.
        # Signal power (output): final_signal_power_w
        # Total noise power in B_ref from all EDFAs, referred to receiver:
        # Each EDFA adds (G-1)NF h nu B_ref. If G compensates span loss, this noise is added to signal at that point.
        # Sum of noise powers, each amplified by subsequent gains.
        # Simplified: Assume OSNR is degraded by NF at each stage roughly.
        # OSNR_input_link_db - N_spans * NF_db (very rough)
        # Let initial OSNR be very high. Then OSNR_final_dB = OSNR_initial_dB - sum_of_NF_impacts
        # OSNR_final_linear = final_signal_power_w / total_accumulated_ase_power_in_ref_bw_w
        # This is where a full noise model gets complex. The per-EDFA noise added previously is one way.
        # Let's stick to the noise added by EDFAs. The `target_osnr_db` is an input, let's use it as a floor if our calculated is worse.

        output_osnr_linear = final_signal_power_w / total_ase_power_in_bw_ref_w if total_ase_power_in_bw_ref_w > 0 else float('inf')
        output_osnr_db = 10 * np.log10(output_osnr_linear) if output_osnr_linear > 0 else -float('inf')
        log_details.append(f"Estimated OSNR at receiver from accumulated ASE: {output_osnr_db:.2f} dB (in {bw_ref_nm}nm BW)")

        # If the calculated OSNR from ASE is worse than the input target_osnr_db, 
        # it implies other unmodeled noise. We can add more noise to meet the target_osnr_db 
        # OR report the calculated one. Let's report the calculated one due to component effects.
        # If no EDFAs (spans=0), then target_osnr_db must be achieved by adding receiver noise.
        if spans == 0:
            # Add receiver noise to meet target_osnr_db
            signal_power_linear = final_signal_power_w # Current signal power
            target_osnr_linear_val = 10**(target_osnr_db/10)
            noise_power_for_target_osnr_w = signal_power_linear / target_osnr_linear_val
            # Add complex Gaussian noise based on this noise power (scaled for normalized symbols)
            # Again, assuming symbols are normalized to unit average power for simplicity of noise scaling here.
            # Variance = noise_power_for_target_osnr_w / (2 * symbol_power_reference_for_normalization)
            # If current_symbols are normalized, noise_std relates to 1/sqrt(OSNR_linear)
            noise_std_dev_rx = np.sqrt(1 / (2 * target_osnr_linear_val)) # Per quadrature for normalized symbols
            real_noise_rx = np.random.normal(0, noise_std_dev_rx, len(current_symbols))
            imag_noise_rx = np.random.normal(0, noise_std_dev_rx, len(current_symbols))
            current_symbols += (real_noise_rx + 1j * imag_noise_rx)
            log_details.append(f"No spans. Added receiver noise to meet target OSNR of {target_osnr_db:.2f} dB. Noise std dev ~{noise_std_dev_rx:.3f}")
            output_osnr_db = target_osnr_db # By definition for this case
            output_osnr_linear = 10**(output_osnr_db/10)
        
        # --- DSP for CD and SPM (simplified) ---
        if apply_dsp:
            # Ideal CD compensation (removes accumulated linear phase distortion - not fully modeled above as phase on symbols)
            log_details.append("Applied DSP: Ideal CD compensation.")
            # Ideal SPM compensation (removes common nonlinear phase - also not fully modeled above as phase on symbols)
            # This is a gross simplification. Real DSP is complex.
            # We can attempt to rotate back the mean nonlinear phase if it was globally applied.
            # current_symbols *= np.exp(-1j * accumulated_nonlinear_phase_rad) # If it was a common phase shift
            log_details.append("Applied DSP: Simplified common NL phase compensation.")
            # DSP might also try to correct phase noise / linewidth effects (e.g. using a PLL)
            # And perform symbol synchronization, equalization, etc.
            # For now, just log that DSP is active.

        propagation_summary = {
            "output_power_receiver_dbm": 10 * np.log10(final_signal_power_w * 1000) if final_signal_power_w > 0 else -float('inf'),
            "total_attenuation_db": total_attenuation_db,
            "accumulated_dispersion_ps_nm": accumulated_dispersion_ps_nm,
            "accumulated_nonlinear_phase_rad": accumulated_nonlinear_phase_rad,
            "estimated_output_osnr_db": output_osnr_db,
            "estimated_output_osnr_linear": output_osnr_linear,
            "log_details": log_details
        }

        return current_symbols, propagation_summary

    def _generate_random_symbols(self, num_symbols: int, constellation: np.ndarray) -> np.ndarray:
        """Generate random symbols based on the given constellation"""
        return np.random.choice(constellation, num_symbols)

    def _calculate_snr(self, symbols: np.ndarray, received_symbols: np.ndarray) -> float:
        """Calculate SNR based on the given symbols and received symbols"""
        # Implement SNR calculation logic here
        return 10.0  # Placeholder return, actual implementation needed

    def _calculate_evm(self, received_symbols: np.ndarray, constellation: np.ndarray) -> float:
        """Calculate EVM based on the given received symbols and constellation"""
        # Implement EVM calculation logic here
        return 0.0  # Placeholder return, actual implementation needed

    def _calculate_theoretical_ber(self, snr_linear: float, modulation: str) -> float:
        """Calculate theoretical BER based on the given SNR and modulation"""
        # Implement BER calculation logic here
        return 0.0  # Placeholder return, actual implementation needed

    def _estimate_reach(self, snr_db: float, modulation: str) -> float:
        """Estimate reach based on the given SNR and modulation"""
        # Implement reach estimation logic here
        return 100.0  # Placeholder return, actual implementation needed