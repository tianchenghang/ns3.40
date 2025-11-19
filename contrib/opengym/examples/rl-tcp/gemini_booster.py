#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
import joblib
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import threading
from collections import deque
import logging

class GeminiBooster:
    """
    Booster: Online optimization engine for Gemini congestion control
    Based on Bayesian Optimization with hidden variables
    """

    def __init__(self, config_file: str = None):
        # Default configuration
        self.config = {
            'parameter_bounds': {
                'alpha': (1.0, 3.0),
                'gamma': (1.0, 6.0),
                'lambda': (0.5, 0.95),
                'loss_thresh': (0.001, 0.05),
                'rtt_thresh': (0.1, 0.8),
                'window_size': (1, 10)
            },
            'hidden_variables': ['region', 'isp', 'time_of_day'],
            'optimization': {
                'acquisition_function': 'ei',  # Expected Improvement
                'exploration_weight': 0.1,
                'max_iterations': 100,
                'convergence_threshold': 0.01
            },
            'performance': {
                'utility_function': 'throughput - 0.1 * delay',
                'sigma': 0.1
            }
        }

        if config_file:
            self.load_config(config_file)

        # Initialize optimization components
        self.history = deque(maxlen=1000)
        self.surrogate_model = None
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_trained = False

        # Hidden variable mappings
        self.region_map = {'east': 0, 'west': 1, 'north': 2, 'south': 3}
        self.isp_map = {'telecom': 0, 'unicom': 1, 'mobile': 2}

        # Thread safety
        self.lock = threading.Lock()

        # Setup logging
        self.setup_logging()

        self.logger.info("Gemini Booster initialized")

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('gemini_booster.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('GeminiBooster')

    def load_config(self, config_file: str):
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                user_config = json.load(f)

            # Deep merge configurations
            for key, value in user_config.items():
                if isinstance(value, dict) and key in self.config:
                    self.config[key].update(value)
                else:
                    self.config[key] = value

            self.logger.info(f"Configuration loaded from {config_file}")
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")

    def encode_hidden_variables(self, hidden_vars: Dict) -> List[float]:
        """Encode hidden variables to numerical features"""
        encoded = []

        # Region (one-hot encoding)
        region_encoded = [0] * len(self.region_map)
        if hidden_vars.get('region') in self.region_map:
            region_encoded[self.region_map[hidden_vars['region']]] = 1
        encoded.extend(region_encoded)

        # ISP (one-hot encoding)
        isp_encoded = [0] * len(self.isp_map)
        if hidden_vars.get('isp') in self.isp_map:
            isp_encoded[self.isp_map[hidden_vars['isp']]] = 1
        encoded.extend(isp_encoded)

        # Time of day (cyclic encoding)
        time_of_day = hidden_vars.get('time_of_day', 12)
        encoded.append(np.sin(2 * np.pi * time_of_day / 24))
        encoded.append(np.cos(2 * np.pi * time_of_day / 24))

        return encoded

    def calculate_utility(self, performance: Dict) -> float:
        """Calculate utility function from performance metrics"""
        throughput = performance.get('throughput', 0)
        delay = performance.get('delay', 0)  # in ms
        loss_rate = performance.get('loss_rate', 0)

        # Basic utility: throughput - σ * delay
        sigma = self.config['performance']['sigma']
        utility = throughput - sigma * delay

        # Penalize high loss rates
        if loss_rate > 0.1:  # 10% threshold
            utility *= (1 - loss_rate)

        return utility

    def add_sample(self, parameters: Dict, hidden_vars: Dict, performance: Dict):
        """Add a new sample to the history"""
        with self.lock:
            # Encode features
            param_vector = [
                parameters['alpha'],
                parameters['gamma'],
                parameters['lambda'],
                parameters['loss_thresh'],
                parameters['rtt_thresh'],
                parameters['window_size']
            ]

            hidden_vector = self.encode_hidden_variables(hidden_vars)
            feature_vector = param_vector + hidden_vector

            # Calculate utility
            utility = self.calculate_utility(performance)

            # Store sample
            sample = {
                'timestamp': datetime.now(),
                'features': feature_vector,
                'parameters': parameters.copy(),
                'hidden_vars': hidden_vars.copy(),
                'performance': performance.copy(),
                'utility': utility
            }

            self.history.append(sample)
            self.logger.debug(f"Added sample with utility: {utility:.4f}")

            # Retrain model periodically
            if len(self.history) % 50 == 0:
                self.train_model()

    def train_model(self):
        """Train the surrogate model using historical data"""
        if len(self.history) < 10:
            self.logger.warning("Insufficient data for training")
            return

        with self.lock:
            try:
                # Prepare training data
                features = []
                utilities = []

                for sample in self.history:
                    features.append(sample['features'])
                    utilities.append(sample['utility'])

                X = np.array(features)
                y = np.array(utilities).reshape(-1, 1)

                # Scale features
                X_scaled = self.scaler_x.fit_transform(X)
                y_scaled = self.scaler_y.fit_transform(y)

                # Define kernel for Gaussian Process
                kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))

                # Create and train GP model
                self.surrogate_model = GaussianProcessRegressor(
                    kernel=kernel,
                    n_restarts_optimizer=10,
                    alpha=1e-2
                )

                self.surrogate_model.fit(X_scaled, y_scaled.ravel())
                self.is_trained = True

                self.logger.info(f"Surrogate model trained with {len(X)} samples")

            except Exception as e:
                self.logger.error(f"Model training failed: {e}")

    def expected_improvement(self, X: np.ndarray, xi: float = 0.01) -> np.ndarray:
        """Calculate Expected Improvement acquisition function"""
        if not self.is_trained:
            return np.zeros(X.shape[0])

        X_scaled = self.scaler_x.transform(X)

        try:
            y_pred, sigma = self.surrogate_model.predict(X_scaled, return_std=True)
            y_pred = self.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

            # Find best current utility
            best_utility = max(sample['utility'] for sample in self.history)

            # Calculate improvement
            with np.errstate(divide='warn'):
                imp = y_pred - best_utility - xi
                Z = imp / sigma
                ei = imp * self._norm_cdf(Z) + sigma * self._norm_pdf(Z)
                ei[sigma == 0.0] = 0.0

            return ei

        except Exception as e:
            self.logger.error(f"EI calculation failed: {e}")
            return np.zeros(X.shape[0])

    def _norm_cdf(self, x):
        """Cumulative distribution function for standard normal"""
        return (1.0 + np.erf(x / np.sqrt(2.0))) / 2.0

    def _norm_pdf(self, x):
        """Probability density function for standard normal"""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

    def optimize_parameters(self, hidden_vars: Dict) -> Dict:
        """Optimize parameters for given hidden variables"""
        if not self.is_trained or len(self.history) < 20:
            return self.get_default_parameters()

        self.logger.info(f"Optimizing parameters for hidden vars: {hidden_vars}")

        best_params = None
        best_ei = -np.inf

        # Generate candidate parameters
        n_candidates = 1000
        candidates = self._generate_candidates(n_candidates, hidden_vars)

        # Evaluate Expected Improvement
        ei_values = self.expected_improvement(candidates)

        # Find best candidate
        best_idx = np.argmax(ei_values)
        best_candidate = candidates[best_idx]

        # Convert back to parameter dictionary
        best_params = {
            'alpha': best_candidate[0],
            'gamma': best_candidate[1],
            'lambda': best_candidate[2],
            'loss_thresh': best_candidate[3],
            'rtt_thresh': best_candidate[4],
            'window_size': int(round(best_candidate[5]))
        }

        self.logger.info(f"Optimized parameters: {best_params}")
        return best_params

    def _generate_candidates(self, n_candidates: int, hidden_vars: Dict) -> np.ndarray:
        """Generate candidate parameter vectors"""
        bounds = self.config['parameter_bounds']
        hidden_encoded = self.encode_hidden_variables(hidden_vars)

        candidates = []

        for _ in range(n_candidates):
            # Random parameters within bounds
            params = [
                np.random.uniform(bounds['alpha'][0], bounds['alpha'][1]),
                np.random.uniform(bounds['gamma'][0], bounds['gamma'][1]),
                np.random.uniform(bounds['lambda'][0], bounds['lambda'][1]),
                np.random.uniform(bounds['loss_thresh'][0], bounds['loss_thresh'][1]),
                np.random.uniform(bounds['rtt_thresh'][0], bounds['rtt_thresh'][1]),
                np.random.uniform(bounds['window_size'][0], bounds['window_size'][1])
            ]

            # Combine with hidden variables
            candidate = params + hidden_encoded
            candidates.append(candidate)

        return np.array(candidates)

    def get_default_parameters(self) -> Dict:
        """Get default Gemini parameters"""
        bounds = self.config['parameter_bounds']

        return {
            'alpha': np.mean(bounds['alpha']),
            'gamma': np.mean(bounds['gamma']),
            'lambda': np.mean(bounds['lambda']),
            'loss_thresh': np.mean(bounds['loss_thresh']),
            'rtt_thresh': np.mean(bounds['rtt_thresh']),
            'window_size': int(np.mean(bounds['window_size']))
        }

    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.history:
            return {}

        utilities = [sample['utility'] for sample in self.history]
        throughputs = [sample['performance']['throughput'] for sample in self.history]
        delays = [sample['performance']['delay'] for sample in self.history]

        return {
            'samples_count': len(self.history),
            'avg_utility': np.mean(utilities),
            'std_utility': np.std(utilities),
            'avg_throughput': np.mean(throughputs),
            'avg_delay': np.mean(delays),
            'best_utility': max(utilities) if utilities else 0
        }

    def save_model(self, filepath: str):
        """Save the trained model to disk"""
        if not self.is_trained:
            self.logger.warning("No trained model to save")
            return

        try:
            model_data = {
                'surrogate_model': self.surrogate_model,
                'scaler_x': self.scaler_x,
                'scaler_y': self.scaler_y,
                'history': list(self.history),
                'config': self.config
            }

            joblib.dump(model_data, filepath)
            self.logger.info(f"Model saved to {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")

    def load_model(self, filepath: str):
        """Load a trained model from disk"""
        try:
            model_data = joblib.load(filepath)

            self.surrogate_model = model_data['surrogate_model']
            self.scaler_x = model_data['scaler_x']
            self.scaler_y = model_data['scaler_y']
            self.history = deque(model_data['history'], maxlen=1000)
            self.config = model_data['config']
            self.is_trained = True

            self.logger.info(f"Model loaded from {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")

class GeminiAgent:
    """
    Optimization Agent: Connects Gemini Fusion with Booster
    """

    def __init__(self, booster: GeminiBooster, agent_id: str = "default"):
        self.booster = booster
        self.agent_id = agent_id
        self.current_parameters = booster.get_default_parameters()
        self.performance_history = []

        # Hidden variables for this agent
        self.hidden_vars = {
            'region': 'east',
            'isp': 'telecom',
            'time_of_day': 12
        }

        self.logger = logging.getLogger(f'GeminiAgent_{agent_id}')
        self.logger.info(f"Gemini Agent {agent_id} initialized")

    def update_hidden_variables(self, region: str = None, isp: str = None,
                              time_of_day: int = None):
        """Update hidden variables for this agent"""
        if region:
            self.hidden_vars['region'] = region
        if isp:
            self.hidden_vars['isp'] = isp
        if time_of_day is not None:
            self.hidden_vars['time_of_day'] = time_of_day % 24

    def report_performance(self, performance_metrics: Dict):
        """Report performance metrics to Booster"""
        try:
            # Add sample to booster history
            self.booster.add_sample(
                parameters=self.current_parameters,
                hidden_vars=self.hidden_vars,
                performance=performance_metrics
            )

            # Store in local history
            self.performance_history.append({
                'timestamp': datetime.now(),
                'parameters': self.current_parameters.copy(),
                'performance': performance_metrics.copy(),
                'utility': self.booster.calculate_utility(performance_metrics)
            })

            self.logger.debug(f"Performance reported: {performance_metrics}")

        except Exception as e:
            self.logger.error(f"Failed to report performance: {e}")

    def get_optimized_parameters(self) -> Dict:
        """Get optimized parameters from Booster"""
        try:
            optimized_params = self.booster.optimize_parameters(self.hidden_vars)
            self.current_parameters = optimized_params

            self.logger.info(f"New parameters received: {optimized_params}")
            return optimized_params

        except Exception as e:
            self.logger.error(f"Failed to get optimized parameters: {e}")
            return self.booster.get_default_parameters()

    def get_performance_summary(self) -> Dict:
        """Get performance summary for this agent"""
        if not self.performance_history:
            return {}

        recent_history = self.performance_history[-50:]  # Last 50 samples

        throughputs = [p['performance']['throughput'] for p in recent_history]
        delays = [p['performance']['delay'] for p in recent_history]
        utilities = [p['utility'] for p in recent_history]

        return {
            'agent_id': self.agent_id,
            'samples_count': len(recent_history),
            'avg_throughput': np.mean(throughputs),
            'avg_delay': np.mean(delays),
            'avg_utility': np.mean(utilities),
            'current_parameters': self.current_parameters,
            'hidden_vars': self.hidden_vars
        }

# Example usage and testing
def demo_gemini_booster():
    """Demonstrate the Gemini Booster system"""

    # Initialize Booster
    booster = GeminiBooster()

    # Create multiple agents for different network conditions
    agents = {
        'east_telecom': GeminiAgent(booster, 'east_telecom'),
        'west_unicom': GeminiAgent(booster, 'west_unicom'),
        'north_mobile': GeminiAgent(booster, 'north_mobile')
    }

    # Set different hidden variables for each agent
    agents['east_telecom'].update_hidden_variables('east', 'telecom', 9)
    agents['west_unicom'].update_hidden_variables('west', 'unicom', 14)
    agents['north_mobile'].update_hidden_variables('north', 'mobile', 20)

    # Simulate performance reporting and optimization
    print("Starting Gemini Booster demonstration...")

    for iteration in range(10):
        print(f"\n--- Iteration {iteration + 1} ---")

        for agent_name, agent in agents.items():
            # Simulate performance metrics (in real scenario, these come from ns-3)
            performance = {
                'throughput': np.random.uniform(10, 100),
                'delay': np.random.uniform(10, 100),
                'loss_rate': np.random.uniform(0, 0.05)
            }

            # Report performance
            agent.report_performance(performance)

            # Get optimized parameters (every 3 iterations)
            if iteration % 3 == 0:
                new_params = agent.get_optimized_parameters()
                print(f"{agent_name}: New parameters -> {new_params}")

            # Print performance summary
            summary = agent.get_performance_summary()
            print(f"{agent_name}: Avg throughput={summary['avg_throughput']:.2f}Mbps, "
                  f"delay={summary['avg_delay']:.2f}ms, utility={summary['avg_utility']:.2f}")

    # Print overall booster statistics
    booster_stats = booster.get_performance_stats()
    print(f"\n--- Booster Statistics ---")
    print(f"Total samples: {booster_stats['samples_count']}")
    print(f"Average utility: {booster_stats['avg_utility']:.2f}")
    print(f"Best utility: {booster_stats['best_utility']:.2f}")

    # Save model
    booster.save_model('gemini_booster_model.pkl')

if __name__ == "__main__":
    demo_gemini_booster()
