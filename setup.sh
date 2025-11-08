#!/bin/bash

echo "Setting up SmartWaste AI Food Waste Reduction System..."

# Create necessary directories
mkdir -p .streamlit
mkdir -p data
mkdir -p utils
mkdir -p assets

echo "📁 Directories created successfully!"

# Create sample data file
cat > data/sample_data.csv << 'EOF'
name,food_type,quantity_kg,days_to_expiry,storage_temp,price_per_kg
Organic Apples,Fruits,2.5,7,4,120
Fresh Carrots,Vegetables,1.2,14,4,60
Whole Milk,Dairy,1.0,5,4,60
Chicken Breast,Meat,1.5,3,2,300
Basmati Rice,Grains,5.0,90,25,80
Whole Wheat Bread,Bakery,0.5,2,25,40
Bananas,Fruits,1.8,3,25,40
Spinach,Vegetables,0.8,4,4,80
Yogurt,Dairy,0.5,10,4,50
Salmon Fillet,Meat,1.0,4,2,500
EOF

echo "📊 Sample data created!"

# Create CSS file
cat > assets/style.css << 'EOF'
/* SmartWaste Custom Styles */
.main-header {
    background: linear-gradient(135deg, #2ecc71 0%, #3498db 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 2rem;
    font-size: 2.5rem;
}

.metric-card {
    background: white;
    border-radius: 15px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    border-left: 5px solid #2ecc71;
    transition: transform 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
}

.waste-saved {
    border-left-color: #2ecc71;
    background: linear-gradient(135deg, #d4edda 0%, #c8e6c9 100%);
}

.waste-alert {
    border-left-color: #e74c3c;
    background: linear-gradient(135deg, #f8d7da 0%, #ffcdd2 100%);
    animation: pulse 2s infinite;
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
}

.recommendation {
    background: linear-gradient(135deg, #d1ecf1 0%, #b3e5fc 100%);
    border-left: 5px solid #3498db;
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
    transition: all 0.3s ease;
}

.recommendation:hover {
    transform: translateX(5px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

.food-type-fruits { border-left-color: #e74c3c; background-color: #ffebee; }
.food-type-vegetables { border-left-color: #2ecc71; background-color: #e8f5e8; }
.food-type-dairy { border-left-color: #f39c12; background-color: #fff3e0; }
.food-type-meat { border-left-color: #c0392b; background-color: #fbe9e7; }
.food-type-grains { border-left-color: #8e44ad; background-color: #f3e5f5; }
.food-type-bakery { border-left-color: #d35400; background-color: #fff8e1; }

/* Animations */
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(231, 76, 60, 0.4); }
    70% { box-shadow: 0 0 0 10px rgba(231, 76, 60, 0); }
    100% { box-shadow: 0 0 0 0 rgba(231, 76, 60, 0); }
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.fade-in {
    animation: fadeIn 0.6s ease-in;
}

/* Progress bars */
.stProgress > div > div > div > div {
    background-color: #2ecc71;
}

/* Custom buttons */
.stButton > button {
    background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-weight: 600;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(46, 204, 113, 0.3);
}

/* Risk indicators */
.risk-low { color: #27ae60; }
.risk-medium { color: #f39c12; }
.risk-high { color: #e74c3c; }

/* Loading animation */
.loading-dots:after {
    content: '...';
    animation: dots 1.5s steps(4, end) infinite;
}

@keyframes dots {
    0%, 20% { color: rgba(0,0,0,0); text-shadow: .25em 0 0 rgba(0,0,0,0), .5em 0 0 rgba(0,0,0,0); }
    40% { color: #2ecc71; text-shadow: .25em 0 0 rgba(0,0,0,0), .5em 0 0 rgba(0,0,0,0); }
    60% { text-shadow: .25em 0 0 #2ecc71, .5em 0 0 rgba(0,0,0,0); }
    80%, 100% { text-shadow: .25em 0 0 #2ecc71, .5em 0 0 #2ecc71; }
}
EOF

echo "🎨 CSS styles created!"

# Create utils files
cat > utils/__init__.py << 'EOF'
# Utils package for SmartWaste
EOF

cat > utils/waste_predictor.py << 'EOF'
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

class WastePredictor:
    def __init__(self):
        self.model = None
        self.features = ['quantity_kg', 'days_to_expiry', 'food_type', 'storage_temp', 'price_per_kg']
        self.food_types = {
            'Fruits': 0, 'Vegetables': 1, 'Dairy': 2, 
            'Meat': 3, 'Grains': 4, 'Bakery': 5
        }
    
    def prepare_data(self):
        """Generate sample training data"""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'quantity_kg': np.random.uniform(0.1, 10, n_samples),
            'days_to_expiry': np.random.randint(1, 30, n_samples),
            'food_type': np.random.choice(list(self.food_types.keys()), n_samples),
            'storage_temp': np.random.uniform(-5, 25, n_samples),
            'price_per_kg': np.random.uniform(5, 500, n_samples),
        }
        
        # Simulate waste percentage
        waste_base = (
            data['days_to_expiry'] * 0.1 +
            (30 - data['days_to_expiry']) * 0.3 +
            np.abs(data['storage_temp'] - 4) * 0.2 +
            data['quantity_kg'] * 0.05
        )
        
        data['waste_percentage'] = np.clip(waste_base + np.random.normal(0, 5, n_samples), 0, 100)
        data['waste_kg'] = data['quantity_kg'] * data['waste_percentage'] / 100
        
        return pd.DataFrame(data)
    
    def train_model(self):
        """Train the waste prediction model"""
        df = self.prepare_data()
        X = df[self.features].copy()
        X['food_type'] = X['food_type'].map(self.food_types)
        y = df['waste_percentage']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        return self.model
EOF

cat > utils/image_processor.py << 'EOF'
import numpy as np
from PIL import Image

class ImageProcessor:
    def __init__(self):
        self.food_classes = ['Fruits', 'Vegetables', 'Dairy', 'Meat', 'Grains', 'Bakery']
    
    def analyze_image(self, image):
        """Simulate image analysis for food recognition"""
        try:
            img = Image.open(image)
            img_array = np.array(img)
            
            if len(img_array.shape) == 3:
                avg_color = np.mean(img_array, axis=(0, 1))
                
                if avg_color[1] > 150:
                    detected_food = 'Vegetables'
                elif avg_color[0] > 150:
                    detected_food = 'Fruits'
                elif avg_color[2] > 150:
                    detected_food = 'Dairy'
                else:
                    detected_food = 'Bakery'
                    
                confidence = np.random.uniform(0.7, 0.95)
            else:
                detected_food = 'Unknown'
                confidence = 0.0
                
            return {
                'detected_food': detected_food,
                'confidence': confidence,
                'message': f"Detected: {detected_food} (Confidence: {confidence:.1%})"
            }
            
        except Exception as e:
            return {
                'detected_food': 'Unknown',
                'confidence': 0.0,
                'message': f"Error analyzing image: {str(e)}"
            }
EOF

echo "🔧 Utility files created!"

# Create Streamlit config
cat > .streamlit/config.toml << 'EOF'
[server]
headless = true
port = 8501
enableCORS = false

[theme]
primaryColor = "#2ecc71"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
EOF

echo "⚙️ Streamlit config created!"

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "🎉 SmartWaste setup completed successfully!"
echo ""
echo "🚀 To run the application:"
echo "   streamlit run app.py"
echo ""
echo "📱 Then open http://localhost:8501 in your browser"
echo ""
echo "🌱 Start reducing food waste with AI!"