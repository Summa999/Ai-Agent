import yaml
import json
import os
from typing import Dict, Any, Optional
from pathlib import Path

class ConfigManager:
    def __init__(self, config_path: str = 'config.yaml'):
        self.config_path = Path(config_path)
        self.config = {}
        self.load_config()
    
    def load_config(self):
        """Load configuration from file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        # Load main config
        with open(self.config_path, 'r') as f:
            if self.config_path.suffix == '.yaml':
                self.config = yaml.safe_load(f)
            elif self.config_path.suffix == '.json':
                self.config = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {self.config_path.suffix}")
        
        # Load environment-specific overrides
        self._load_env_config()
        
        # Load secrets from environment variables
        self._load_secrets()
    
    def _load_env_config(self):
        """Load environment-specific configuration"""
        env = os.getenv('TRADING_ENV', 'development')
        env_config_path = self.config_path.parent / f"config.{env}.yaml"
        
        if env_config_path.exists():
            with open(env_config_path, 'r') as f:
                env_config = yaml.safe_load(f)
                
            # Deep merge configs
            self.config = self._deep_merge(self.config, env_config)
    
    def _load_secrets(self):
        """Load secrets from environment variables"""
        # MT5 credentials
        if 'MT5_ACCOUNT' in os.environ:
            self.config['mt5']['account'] = int(os.environ['MT5_ACCOUNT'])
        if 'MT5_PASSWORD' in os.environ:
            self.config['mt5']['password'] = os.environ['MT5_PASSWORD']
        
        # API keys
        api_keys = self.config.get('api_keys', {})
        if 'ALPHA_VANTAGE_KEY' in os.environ:
            api_keys['alpha_vantage'] = os.environ['ALPHA_VANTAGE_KEY']
        if 'NEWSAPI_KEY' in os.environ:
            api_keys['newsapi'] = os.environ['NEWSAPI_KEY']
        
        self.config['api_keys'] = api_keys
        
        # Telegram
        if 'TELEGRAM_BOT_TOKEN' in os.environ:
            self.config['telegram']['bot_token'] = os.environ['TELEGRAM_BOT_TOKEN']
        if 'TELEGRAM_CHAT_ID' in os.environ:
            self.config['telegram']['chat_id'] = os.environ['TELEGRAM_CHAT_ID']
        
        # Database
        if 'DB_PASSWORD' in os.environ:
            self.config['database']['password'] = os.environ['DB_PASSWORD']
    
    def _deep_merge(self, dict1: Dict, dict2: Dict) -> Dict:
        """Deep merge two dictionaries"""
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get config value using dot notation"""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any):
        """Set config value using dot notation"""
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def save(self, path: Optional[str] = None):
        """Save configuration to file"""
        save_path = Path(path) if path else self.config_path
        
        with open(save_path, 'w') as f:
            if save_path.suffix == '.yaml':
                yaml.dump(self.config, f, default_flow_style=False)
            elif save_path.suffix == '.json':
                json.dump(self.config, f, indent=2)
    
    def validate(self) -> bool:
        """Validate configuration"""
        required_keys = [
            'mt5.account',
            'mt5.password',
            'mt5.server',
            'trading.symbols',
            'risk.max_risk_per_trade',
            'telegram.bot_token',
            'telegram.chat_id'
        ]
        
        for key in required_keys:
            if self.get(key) is None:
                print(f"Missing required configuration: {key}")
                return False
        
        return True