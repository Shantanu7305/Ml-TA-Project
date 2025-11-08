import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

class WastePredictor:
    def __init__(self):
        self.model = None
        self.features = ['quantity_kg', 'days_to_expiry', 'food_type', 'storage_temp', 'price_per_kg']
        
    def train_model(self, data):
        X = data[self.features]
        y = data['waste_percentage']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        return self.model