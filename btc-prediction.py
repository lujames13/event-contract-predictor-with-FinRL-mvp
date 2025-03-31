"""
Binance BTC/USDT Price Movement Prediction using FinRL

This script adapts the FinRL library to create price movement predictions (up/down)
for BTC/USDT trading pair at different time horizons (10m, 30m, 1h, 1d).

Requirements:
- FinRL library 
- Python-binance package
- dotenv package for environment variables
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings("ignore")

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Binance API
from binance.client import Client
from binance.exceptions import BinanceAPIException

# Import FinRL components
from finrl.config import INDICATORS
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline

# Create necessary directories
import os
from finrl.main import check_and_make_directories
from finrl.config import (
    DATA_SAVE_DIR,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
    RESULTS_DIR,
)

check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])


class BinanceDataDownloader:
    """Class to download historical data from Binance"""
    
    def __init__(self):
        # Initialize Binance client using API keys from environment variables
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        
        print("Checking Binance API credentials...")
        
        if not api_key or not api_secret:
            raise ValueError("Binance API credentials not found in environment variables")
        else:
            print("API credentials found in environment variables")
            
        # Create client
        self.client = Client(api_key, api_secret)
        
        # Test API connection
        try:
            server_time = self.client.get_server_time()
            print(f"Connected to Binance API. Server time: {datetime.fromtimestamp(server_time['serverTime']/1000)}")
        except Exception as e:
            print(f"Warning: Could not connect to Binance API: {e}")
            
        # Create data storage directory if it doesn't exist
        self.data_dir = os.path.join('data', 'binance')
        os.makedirs(self.data_dir, exist_ok=True)
            
    def get_historical_klines(self, symbol, interval, start_str, end_str=None, use_cache=True):
        """
        Get historical klines from Binance
        
        Parameters:
        -----------
        symbol : str
            Trading pair symbol (e.g., 'BTCUSDT')
        interval : str
            Kline interval (e.g., '1m', '5m', '1h', '1d')
        start_str : str
            Start date string in "YYYY-MM-DD" format
        end_str : str, optional
            End date string in "YYYY-MM-DD" format
        use_cache : bool, optional
            Whether to use cached data if available
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing historical price data
        """
        # Convert dates to datetime objects for easier handling
        start_date = pd.to_datetime(start_str)
        end_date = pd.to_datetime(end_str) if end_str else pd.Timestamp.now()
        
        # Generate a filename for this data request
        cache_filename = f"{symbol}_{interval}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
        cache_path = os.path.join(self.data_dir, cache_filename)
        
        # Check if we have cached data
        if use_cache and os.path.exists(cache_path):
            print(f"Loading cached data from {cache_path}")
            data = pd.read_csv(cache_path)
            
            # Convert date column to datetime
            data['date'] = pd.to_datetime(data['date'])
            
            print(f"Loaded {len(data)} data points from cache")
            return data
            
        try:
            print(f"Requesting historical klines for {symbol}, interval {interval}")
            print(f"From {start_str} to {end_str}")
            
            # Get klines from Binance
            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_str,
                end_str=end_str
            )
            
            print(f"Received {len(klines)} klines from Binance API")
            
            if len(klines) == 0:
                print("Warning: No data returned from Binance API")
                return None
                
            # Convert to DataFrame
            data = pd.DataFrame(
                klines,
                columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ]
            )
            
            # Convert timestamp to datetime
            data['date'] = pd.to_datetime(data['timestamp'], unit='ms')
            
            # Convert numeric columns to float
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
                              'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
            data[numeric_columns] = data[numeric_columns].astype(float)
            
            # Add symbol column
            data['tic'] = symbol
            
            # Select relevant columns
            data = data[['date', 'tic', 'open', 'high', 'low', 'close', 'volume']]
            
            # Cache the data to disk
            data.to_csv(cache_path, index=False)
            print(f"Cached data to {cache_path}")
            
            return data
            
        except BinanceAPIException as e:
            print(f"Binance API error: {e}")
            return None
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return None


class PriceMovementPredictor:
    """Class to predict price movements using FinRL"""
    
    def __init__(self, timeframes=None):
        """
        Initialize the price movement predictor
        
        Parameters:
        -----------
        timeframes : list, optional
            List of timeframes to predict (in minutes)
        """
        # Default timeframes: 10min, 30min, 1h, 1d
        self.timeframes = timeframes or [10, 30, 60, 1440]
        
        # Generate timeframe names from the timeframes list
        self.timeframe_names = []
        for tf in self.timeframes:
            if tf == 10:
                self.timeframe_names.append('10m')
            elif tf == 30:
                self.timeframe_names.append('30m')
            elif tf == 60:
                self.timeframe_names.append('1h')
            elif tf == 1440:
                self.timeframe_names.append('1d')
            else:
                self.timeframe_names.append(f'{tf}m')
        
    
        # Initialize data downloader
        self.downloader = BinanceDataDownloader()
        
        # Technical indicators to use
        self.indicators = [
            'macd', 'rsi_30', 'cci_30', 'dx_30'
        ]
        
        # Environment parameters
        self.env_kwargs = {
            "hmax": 1,  # Maximum number of shares to trade (1 for binary prediction)
            "initial_amount": 1000,  # Initial capital
            "buy_cost_pct": [0.001],  # 0.1% trading fee (as list)
            "sell_cost_pct": [0.001],  # 0.1% trading fee (as list)
            "reward_scaling": 1e-3,  # Reward scaling factor
            "tech_indicator_list": self.indicators,
            "num_stock_shares": [0],  # Initial number of shares for each stock
        }
        
        # Models to train
        self.models_to_train = ['a2c', 'ppo', 'ddpg', 'sac', 'td3']
        self.model_kwargs = {
            'a2c': {
                'n_steps': 5,
                'ent_coef': 0.005,
                'learning_rate': 0.0007
            },
            'ppo': {
                'ent_coef': 0.01,
                'n_steps': 2048,
                'learning_rate': 0.00025,
                'batch_size': 128
            },
            'ddpg': {
                'buffer_size': 10000,
                'learning_rate': 0.0005,
                'batch_size': 64
            },
            'sac': {
                'batch_size': 64,
                'buffer_size': 100000,
                'learning_rate': 0.0001,
                'learning_starts': 100,
                'ent_coef': 'auto_0.1'
            },
            'td3': {
                'batch_size': 100,
                'buffer_size': 1000000,
                'learning_rate': 0.0001
            }
        }
        self.timesteps_dict = {
            'a2c': 10_000,
            'ppo': 10_000,
            'ddpg': 10_000,
            'sac': 10_000,
            'td3': 10_000
        }
        
    def fetch_and_prepare_data(self, symbol='BTCUSDT', interval='1m', start_date=None, end_date=None, use_cache=True):
        """
        Fetch and prepare data for training and prediction
        
        Parameters:
        -----------
        symbol : str, optional
            Trading pair symbol
        interval : str, optional
            Candlestick interval
        start_date : str, optional
            Start date in 'YYYY-MM-DD' format
        end_date : str, optional
            End date in 'YYYY-MM-DD' format
        use_cache : bool, optional
            Whether to use cached data if available
            
        Returns:
        --------
        tuple
            (processed_data, train_data, test_data)
        """
        # Set default dates if not provided
        if not start_date:
            # Default: 60 days ago
            start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        # Download historical data
        print(f"Downloading {symbol} data from {start_date} to {end_date}...")
        data = self.downloader.get_historical_klines(
            symbol=symbol, 
            interval=interval, 
            start_str=start_date, 
            end_str=end_date,
            use_cache=use_cache
        )
        
        if data is None or len(data) == 0:
            raise ValueError("Failed to fetch data or no data available")
            
        print(f"Downloaded {len(data)} data points.")
        print("Sample data:")
        print(data.head())
        
        # Ensure data has the correct format for FinRL
        # Check if the date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(data['date']):
            data['date'] = pd.to_datetime(data['date'])
        
        # Feature engineering
        fe = FeatureEngineer(
            use_technical_indicator=True,
            tech_indicator_list=self.indicators,
            use_turbulence=True,
            user_defined_feature=False
        )
        
        # Preprocess data
        try:
            print("Computing technical indicators...")
            processed = fe.preprocess_data(data)
            processed = processed.copy()
            processed = processed.fillna(0)
            processed = processed.replace(np.inf, 0)
            
            print("Technical indicators computed successfully")
            print("Available features:", processed.columns.tolist())
            print("Processed data sample:")
            print(processed.head())
            
            # Add target labels for different timeframes
            for i, tf in enumerate(self.timeframes):
                # Calculate price change for each timeframe
                print(f"Creating price change labels for {self.timeframe_names[i]}...")
                processed[f'price_change_{self.timeframe_names[i]}'] = processed['close'].pct_change(tf).shift(-tf)
                
                # Create binary labels (1 for up, 0 for down)
                processed[f'target_{self.timeframe_names[i]}'] = (processed[f'price_change_{self.timeframe_names[i]}'] > 0).astype(int)
                
            # Remove rows with NaN values (due to price change calculation)
            processed = processed.dropna()
            
            # Split data into training and testing sets
            train_size = int(len(processed) * 0.8)
            train_data = processed[:train_size]
            test_data = processed[train_size:]
            
            print(f"Data split: {len(train_data)} training samples, {len(test_data)} testing samples")
            
            # Update environment parameters
            stock_dimension = len(processed.tic.unique())
            state_space = 1 + 2 * stock_dimension + len(self.indicators) * stock_dimension
            
            # Update num_stock_shares to match the number of stocks
            num_stock_shares = [0] * stock_dimension
            # Create cost lists that match the number of stocks
            buy_cost_list = [0.001] * stock_dimension
            sell_cost_list = [0.001] * stock_dimension
            
            self.env_kwargs.update({
                "stock_dim": stock_dimension,
                "state_space": state_space,
                "action_space": stock_dimension,
                "num_stock_shares": num_stock_shares,
                "buy_cost_pct": buy_cost_list,
                "sell_cost_pct": sell_cost_list
            })
            
            return processed, train_data, test_data
            
        except Exception as e:
            print(f"Error during data preprocessing: {e}")
            print("Data columns:", data.columns.tolist())
            print("Data types:", data.dtypes)
            import traceback
            traceback.print_exc()
            raise ValueError(f"Failed to process data: {e}")
    
    def train_models(self, train_data, timeframe_idx=0, force_retrain=False):
        """
        Train models for a specific timeframe
        
        Parameters:
        -----------
        train_data : pd.DataFrame
            Training data
        timeframe_idx : int, optional
            Index of the timeframe to train for
        force_retrain : bool, optional
            Whether to force retraining even if models exist
            
        Returns:
        --------
        dict
            Dictionary of trained models
        """
        timeframe = self.timeframe_names[timeframe_idx]
        print(f"Processing models for {timeframe} timeframe...")
        
        # Check if models already exist for this timeframe
        trained_models = {}
        all_models_exist = True
        
        # Check existing models
        for model_name in self.models_to_train:
            model_path = f"{TRAINED_MODEL_DIR}/{model_name}_{timeframe}"
            if os.path.exists(model_path) and not force_retrain:
                print(f"Found existing {model_name} model for {timeframe}")
                try:
                    # Try to load the model
                    model = DRLAgent.get_model(model_name)
                    model.load(model_path)
                    trained_models[model_name] = model
                except Exception as e:
                    print(f"Error loading existing {model_name} model: {e}")
                    all_models_exist = False
            else:
                all_models_exist = False
        
        # If all models exist and we're not forcing retraining, return the loaded models
        if all_models_exist and not force_retrain:
            print(f"All models for {timeframe} already exist. Skipping training.")
            return trained_models
            
        print(f"Training models for {timeframe} timeframe...")
        
        # Verify that the target column exists
        target_column = f'target_{timeframe}'
        if target_column not in train_data.columns:
            print(f"Warning: Target column '{target_column}' not found in data")
            print(f"Available columns: {train_data.columns.tolist()}")
            print("This can happen if you're trying to train for timeframes that weren't in the initialization")
            print(f"Initialized timeframes: {self.timeframe_names}")
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Create a copy of the data with the specific target
        train_data_with_target = train_data.copy()
        
        # Reset index to use integers starting from 0
        train_data_with_target = train_data_with_target.reset_index(drop=True)
        
        # Add day column if needed
        if 'day' not in train_data_with_target.columns:
            train_data_with_target['day'] = train_data_with_target.index
            
        # Add target as a feature for the model
        train_data_with_target['prediction_target'] = train_data_with_target[target_column]
        
        # Create the training environment
        env_train = StockTradingEnv(df=train_data_with_target, **self.env_kwargs)
        
        # Initialize DRL agent
        agent = DRLAgent(env=env_train)
        
        # Train models that don't exist or need retraining
        for model_name in self.models_to_train:
            if model_name not in trained_models or force_retrain:
                print(f"Training {model_name} model...")
                
                # Get model
                model = agent.get_model(model_name, model_kwargs=self.model_kwargs[model_name])
                
                # Train model
                trained_model = agent.train_model(
                    model=model,
                    tb_log_name=f"{model_name}_{timeframe}",
                    total_timesteps=self.timesteps_dict[model_name]
                )
                
                # Save trained model
                trained_models[model_name] = trained_model
                
                # Save model to disk
                trained_model.save(f"{TRAINED_MODEL_DIR}/{model_name}_{timeframe}")
                
                print(f"{model_name} model trained and saved.")
            
        return trained_models
    
    def evaluate_models(self, test_data, trained_models, timeframe_idx=0):
        """
        Evaluate trained models on test data
        
        Parameters:
        -----------
        test_data : pd.DataFrame
            Test data
        trained_models : dict
            Dictionary of trained models
        timeframe_idx : int, optional
            Index of the timeframe to evaluate
            
        Returns:
        --------
        dict
            Dictionary of model performance metrics
        """
        timeframe = self.timeframe_names[timeframe_idx]
        print(f"Evaluating models for {timeframe} timeframe...")
        
        # Create target-specific environment
        target_column = f'target_{timeframe}'
        
        # Create a copy of the data with the specific target
        test_data_with_target = test_data.copy()
        
        # Reset index to use integers starting from 0
        test_data_with_target = test_data_with_target.reset_index(drop=True)
        
        # Add day column if needed
        if 'day' not in test_data_with_target.columns:
            test_data_with_target['day'] = test_data_with_target.index
        
        # Add target as a feature for the model
        test_data_with_target['prediction_target'] = test_data_with_target[target_column]
        
        # Create the test environment
        env_test = StockTradingEnv(df=test_data_with_target, **self.env_kwargs)
        
        # Evaluate each model
        results = {}
        
        for model_name, trained_model in trained_models.items():
            print(f"Evaluating {model_name} model...")
            
            # Evaluate model on test environment
            df_account_value, df_actions = DRLAgent.DRL_prediction(
                model=trained_model,
                environment=env_test
            )
            
            # Calculate performance metrics
            sharpe = (252 ** 0.5) * df_account_value.account_value.pct_change(1).mean() / df_account_value.account_value.pct_change(1).std()
            
            # Calculate accuracy
            df_actions['actual'] = test_data_with_target[target_column].values[:len(df_actions)]
            df_actions['predicted'] = (df_actions['actions'] > 0).astype(int)
            accuracy = (df_actions['actual'] == df_actions['predicted']).mean()
            
            # Store results
            results[model_name] = {
                'sharpe': sharpe,
                'accuracy': accuracy,
                'df_account_value': df_account_value,
                'df_actions': df_actions
            }
            
            print(f"{model_name} evaluation: Sharpe = {sharpe:.4f}, Accuracy = {accuracy:.4f}")
            
        return results
    
    def train_and_evaluate_all(self, symbol='BTCUSDT', interval='1m', start_date=None, end_date=None, use_cache=True, force_retrain=False):
        """
        Train and evaluate models for all timeframes
        
        Parameters:
        -----------
        symbol : str, optional
            Trading pair symbol
        interval : str, optional
            Candlestick interval
        start_date : str, optional
            Start date in 'YYYY-MM-DD' format
        end_date : str, optional
            End date in 'YYYY-MM-DD' format
        use_cache : bool, optional
            Whether to use cached data if available
        force_retrain : bool, optional
            Whether to force retraining even if models exist
            
        Returns:
        --------
        dict
            Dictionary of model performance metrics for all timeframes
        """
        # Fetch and prepare data
        processed, train_data, test_data = self.fetch_and_prepare_data(
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            use_cache=use_cache
        )
        
        all_results = {}
        
        # Train and evaluate for each timeframe
        for i, timeframe in enumerate(self.timeframe_names):
            print(f"\n{'='*50}")
            print(f"Processing {timeframe} timeframe predictions")
            print(f"{'='*50}\n")
            
            try:
                # Train models
                trained_models = self.train_models(train_data, timeframe_idx=i, force_retrain=force_retrain)
                
                # Evaluate models
                results = self.evaluate_models(test_data, trained_models, timeframe_idx=i)
                
                # Store results
                all_results[timeframe] = results
                
            except Exception as e:
                print(f"Error processing {timeframe} timeframe: {e}")
                import traceback
                traceback.print_exc()
                print(f"Skipping {timeframe} timeframe due to errors")
                all_results[timeframe] = {}  # Empty results
                continue
            
        return all_results
    
    def visualize_results(self, all_results):
        """
        Visualize prediction results
        
        Parameters:
        -----------
        all_results : dict
            Results from train_and_evaluate_all method
        """
        # Create visualization for each timeframe
        for timeframe, results in all_results.items():
            print(f"\nVisualizing results for {timeframe} timeframe")
            
            # Skip if no results for this timeframe
            if not results:
                print(f"No results available for {timeframe} timeframe, skipping visualization")
                continue
                
            try:
                # Find the best model based on accuracy
                best_model = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
                print(f"Best model for {timeframe}: {best_model} (Accuracy: {results[best_model]['accuracy']:.4f})")
                
                # Plot account value for best model
                plt.figure(figsize=(12, 6))
                plt.plot(results[best_model]['df_account_value']['account_value'])
                plt.title(f"{best_model.upper()} Model Account Value ({timeframe} predictions)")
                plt.xlabel('Time Steps')
                plt.ylabel('Account Value')
                plt.tight_layout()
                plt.savefig(f"{RESULTS_DIR}/{best_model}_{timeframe}_account_value.png")
                plt.close()
                
                # Plot comparison of all models
                plt.figure(figsize=(12, 6))
                bar_width = 0.35
                x = np.arange(len(results))
                
                # Plot accuracy
                accuracies = [results[model]['accuracy'] for model in results]
                plt.bar(x - bar_width/2, accuracies, bar_width, label='Accuracy')
                
                # Plot normalized Sharpe ratio
                sharpes = [results[model]['sharpe'] for model in results]
                # Handle NaN values in Sharpe ratios
                sharpes = [0 if np.isnan(s) else s for s in sharpes]
                max_sharpe = max([abs(s) for s in sharpes]) if any(sharpes) else 1
                norm_sharpes = [s/max_sharpe * 0.5 + 0.5 for s in sharpes] if max_sharpe > 0 else [0.5 for _ in sharpes]
                plt.bar(x + bar_width/2, norm_sharpes, bar_width, label='Norm. Sharpe')
                
                plt.xlabel('Models')
                plt.ylabel('Metrics')
                plt.title(f'Model Comparison for {timeframe} Predictions')
                plt.xticks(x, list(results.keys()))
                plt.legend()
                plt.tight_layout()
                plt.savefig(f"{RESULTS_DIR}/model_comparison_{timeframe}.png")
                plt.close()
            
            except Exception as e:
                print(f"Error visualizing results for {timeframe}: {e}")
                import traceback
                traceback.print_exc()
        
    def predict_price_movements(self, symbol='BTCUSDT', num_predictions=5, use_cache=True):
        """
        Make price movement predictions for the near future
        
        Parameters:
        -----------
        symbol : str, optional
            Trading pair symbol
        num_predictions : int, optional
            Number of most recent predictions to show
        use_cache : bool, optional
            Whether to use cached data if available
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with price movement predictions
        """
        # Get current price data
        now = datetime.now()
        start_date = (now - timedelta(days=7)).strftime('%Y-%m-%d')
        end_date = now.strftime('%Y-%m-%d')
        
        # Fetch latest data
        data = self.downloader.get_historical_klines(
            symbol=symbol, 
            interval='1m', 
            start_str=start_date, 
            end_str=end_date,
            use_cache=use_cache
        )
        
        if data is None or len(data) == 0:
            raise ValueError("Failed to fetch recent data")
            
        # Feature engineering
        fe = FeatureEngineer(
            use_technical_indicator=True,
            tech_indicator_list=self.indicators,
            use_turbulence=True,
            user_defined_feature=False
        )
        
        # Preprocess data
        processed = fe.preprocess_data(data)
        processed = processed.copy()
        processed = processed.fillna(0)
        processed = processed.replace(np.inf, 0)
        
        # Reset index to use integers starting from 0
        processed = processed.reset_index(drop=True)
        
        # Add day column if needed
        if 'day' not in processed.columns:
            processed['day'] = processed.index
        
        # Get most recent data points
        recent_data = processed.tail(num_predictions)
        
        # Make predictions for each timeframe
        predictions = pd.DataFrame()
        predictions['timestamp'] = recent_data['date']
        predictions['current_price'] = recent_data['close']
        
        # Directory where models are stored
        model_dir = TRAINED_MODEL_DIR
        print(f"Looking for models in: {model_dir}")
        
        for i, timeframe in enumerate(self.timeframe_names):
            # Print available models for debugging
            available_models = [f for f in os.listdir(model_dir) if f.endswith(f"_{timeframe}.zip")]
            print(f"Available models for {timeframe}: {available_models}")
            
            try:
                # Try each model type for this timeframe
                for model_type in ['a2c', 'ppo', 'ddpg', 'sac', 'td3']:
                    model_path = os.path.join(model_dir, f"{model_type}_{timeframe}")
                    
                    # Check if model exists (with or without .zip extension)
                    model_exists = os.path.exists(model_path) or os.path.exists(f"{model_path}.zip")
                    
                    if model_exists:
                        print(f"Found model: {model_type}_{timeframe}")
                        
                        # Add target column
                        processed['prediction_target'] = 0  # Placeholder
                        
                        # Create environment
                        env_pred = StockTradingEnv(df=processed, **self.env_kwargs)
                        
                        # Create a DRLAgent instance first
                        agent = DRLAgent(env=env_pred)
                        
                        # Load model using the agent instance
                        model = agent.get_model(model_name=model_type)
                        
                        # Load the trained weights
                        model.load(model_path)
                        
                        # Make predictions
                        _, df_actions = DRLAgent.DRL_prediction(model=model, environment=env_pred)
                        
                        # Get predictions for recent data points
                        recent_predictions = df_actions.tail(num_predictions)['actions'].values
                        predictions[f"prediction_{timeframe}"] = (recent_predictions > 0).astype(int)
                        
                        # Add predicted direction
                        predictions[f"direction_{timeframe}"] = ["UP" if p > 0 else "DOWN" for p in recent_predictions]
                        
                        # Break after finding the first working model
                        break
                else:
                    # No model found for this timeframe
                    print(f"No trained model found for {timeframe}")
                    predictions[f"prediction_{timeframe}"] = "N/A"
                    predictions[f"direction_{timeframe}"] = "N/A"
                    
            except Exception as e:
                print(f"Error loading model for {timeframe}: {e}")
                import traceback
                traceback.print_exc()
                predictions[f"prediction_{timeframe}"] = "ERROR"
                predictions[f"direction_{timeframe}"] = "ERROR"
                
        return predictions


# Main execution
if __name__ == "__main__":
    try:
        print("Starting BTC/USDT Price Movement Prediction")
        
        # Initialize predictor with only the timeframes we want to use
        predictor = PriceMovementPredictor(timeframes=[10, 30])  # Only 10m and 30m for prediction
        
        # Set a shorter timeframe for initial testing
        test_days = 14  # Use 14 days of data for initial testing
        
        print(f"Using {test_days} days of historical data for initial test")
        
        # Train and evaluate models for timeframes
        start_date = (datetime.now() - timedelta(days=test_days)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"Fetching data from {start_date} to {end_date}")
        
        # Test data download first
        try:
            downloader = BinanceDataDownloader()
            test_data = downloader.get_historical_klines(
                symbol='BTCUSDT',
                interval='1m',
                start_str=start_date,
                end_str=end_date,
                use_cache=True  # Use cached data if available
            )
            
            if test_data is not None and len(test_data) > 0:
                print(f"Successfully downloaded {len(test_data)} data points for testing")
                print("Data columns:", test_data.columns.tolist())
                print("Sample data:")
                print(test_data.head())
            else:
                print("No data received from Binance API")
                raise ValueError("Empty dataset returned from Binance")
                
        except Exception as e:
            print(f"Error during test data download: {e}")
            raise
        
        # If data download successful, proceed with training
        all_results = predictor.train_and_evaluate_all(
            symbol='BTCUSDT',
            interval='1m',
            start_date=start_date,
            end_date=end_date,
            use_cache=True,  # Use cached data if available
            force_retrain=False  # Set to True to force retraining
        )
        
        # Visualize results
        predictor.visualize_results(all_results)
        
        # Make predictions for recent data
        print("\nMaking BTC/USDT price movement predictions...")
        try:
            predictions = predictor.predict_price_movements(
                symbol='BTCUSDT', 
                num_predictions=5,
                use_cache=True
            )
            
            # Print predictions
            print("\nBTC/USDT Price Movement Predictions:")
            print(predictions[['timestamp', 'current_price', 
                              'direction_10m', 'direction_30m']])
        except Exception as e:
            print(f"Error making predictions: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        import traceback
        print(f"Error in main execution: {e}")
        print("Detailed error:")
        traceback.print_exc()