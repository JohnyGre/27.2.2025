
import json
from pathlib import Path

class ConfigManager:
    def __init__(self, config_file='config.json'):
        self.config_file = Path(config_file)
        self.config = self.load_config()

    def load_config(self):
        if not self.config_file.exists():
            raise FileNotFoundError(f"Configuration file {self.config_file} not found.")

        with open(self.config_file, 'r', encoding='utf-8') as file:
            return json.load(file)

    def get(self, *keys, default=None):
        value = self.config
        for key in keys:
            if isinstance(value, dict):  # Kontrola, či je hodnota slovník
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default  # Ak nie je slovník, vráti predvolenú hodnotu
        return value