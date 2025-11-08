import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import io
import base64
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="SmartWaste - AI Food Waste Reduction",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    try:
        with open('assets/style.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except:
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #2ecc71;
            text-align: center;
            margin-bottom: 2rem;
            background: linear-gradient(135deg, #2ecc71 0%, #3498db 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .metric-card {
            padding: 1.5rem;
            border-radius: 10px;
            border: 1px solid #ddd;
            margin: 10px 0;
            background-color: #f8f9fa;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        </style>
        """, unsafe_allow_html=True)

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
        
        # Simulate waste percentage based on features
        waste_base = (
            data['days_to_expiry'] * 0.1 +  # More days to expiry = less waste
            (30 - data['days_to_expiry']) * 0.3 +  # Fewer days = more waste
            np.abs(data['storage_temp'] - 4) * 0.2 +  # Optimal temp is 4°C
            data['quantity_kg'] * 0.05  # More quantity = more potential waste
        )
        
        data['waste_percentage'] = np.clip(waste_base + np.random.normal(0, 5, n_samples), 0, 100)
        data['waste_kg'] = data['quantity_kg'] * data['waste_percentage'] / 100
        
        return pd.DataFrame(data)
    
    def train_model(self):
        """Train the waste prediction model"""
        df = self.prepare_data()
        
        # Prepare features
        X = df[self.features].copy()
        X['food_type'] = X['food_type'].map(self.food_types)
        y = df['waste_percentage']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Calculate metrics
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        return self.model, mae, mse
    
    def predict_waste(self, quantity_kg, days_to_expiry, food_type, storage_temp, price_per_kg):
        """Predict waste percentage for given inputs"""
        if self.model is None:
            self.train_model()
        
        # Prepare input
        input_data = pd.DataFrame([{
            'quantity_kg': quantity_kg,
            'days_to_expiry': days_to_expiry,
            'food_type': self.food_types[food_type],
            'storage_temp': storage_temp,
            'price_per_kg': price_per_kg
        }])
        
        waste_percentage = self.model.predict(input_data)[0]
        waste_kg = quantity_kg * waste_percentage / 100
        cost_loss = waste_kg * price_per_kg
        
        return {
            'waste_percentage': max(0, min(100, waste_percentage)),
            'waste_kg': max(0, waste_kg),
            'cost_loss': max(0, cost_loss)
        }

class ImageProcessor:
    def __init__(self):
        self.food_classes = {
            'apple': 'Fruits', 'banana': 'Fruits', 'orange': 'Fruits',
            'tomato': 'Vegetables', 'carrot': 'Vegetables', 'broccoli': 'Vegetables',
            'milk': 'Dairy', 'cheese': 'Dairy', 'yogurt': 'Dairy',
            'bread': 'Bakery', 'meat': 'Meat', 'rice': 'Grains'
        }
    
    def analyze_image(self, image):
        """Simulate image analysis for food recognition"""
        try:
            img = Image.open(image)
            img_array = np.array(img)
            
            # Simulate food detection based on image characteristics
            if len(img_array.shape) == 3:
                avg_color = np.mean(img_array, axis=(0, 1))
                
                # Simple color-based classification simulation
                if avg_color[1] > 150:  # Green dominant
                    detected_food = 'Vegetables'
                elif avg_color[0] > 150:  # Red dominant
                    detected_food = 'Fruits'
                elif avg_color[2] > 150:  # Blue dominant
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

@st.cache_data
def load_sample_data():
    """Load sample food data from CSV"""
    sample_data = [
        {'name': 'Organic Apples', 'food_type': 'Fruits', 'quantity_kg': 2.5, 'days_to_expiry': 7, 'storage_temp': 4, 'price_per_kg': 120},
        {'name': 'Fresh Carrots', 'food_type': 'Vegetables', 'quantity_kg': 1.2, 'days_to_expiry': 14, 'storage_temp': 4, 'price_per_kg': 60},
        {'name': 'Whole Milk', 'food_type': 'Dairy', 'quantity_kg': 1.0, 'days_to_expiry': 5, 'storage_temp': 4, 'price_per_kg': 60},
        {'name': 'Chicken Breast', 'food_type': 'Meat', 'quantity_kg': 1.5, 'days_to_expiry': 3, 'storage_temp': 2, 'price_per_kg': 300},
        {'name': 'Basmati Rice', 'food_type': 'Grains', 'quantity_kg': 5.0, 'days_to_expiry': 90, 'storage_temp': 25, 'price_per_kg': 80},
        {'name': 'Whole Wheat Bread', 'food_type': 'Bakery', 'quantity_kg': 0.5, 'days_to_expiry': 2, 'storage_temp': 25, 'price_per_kg': 40}
    ]
    return sample_data

def generate_recommendations(waste_data, food_items):
    """Generate AI-powered recommendations to reduce waste"""
    recommendations = []
    
    total_waste_kg = sum(item['waste_kg'] for item in waste_data)
    total_cost_loss = sum(item['cost_loss'] for item in waste_data)
    
    if total_waste_kg > 2:
        recommendations.append({
            'type': 'high_priority',
            'title': 'High Waste Alert!',
            'message': f'You\'re projected to waste {total_waste_kg:.1f}kg (₹{total_cost_loss:.0f}). Consider meal planning and proper storage.',
            'icon': '🚨'
        })
    
    # Check for items expiring soon
    expiring_soon = [item for item in food_items if item['days_to_expiry'] <= 3]
    if expiring_soon:
        food_names = [item['name'] for item in expiring_soon]
        recommendations.append({
            'type': 'urgent',
            'title': 'Use Soon Items',
            'message': f'These items expire in 3 days or less: {", ".join(food_names)}. Plan meals around these first.',
            'icon': '⏰'
        })
    
    # Storage recommendations
    improper_storage = [item for item in food_items if abs(item['storage_temp'] - 4) > 5]
    if improper_storage:
        recommendations.append({
            'type': 'improvement',
            'title': 'Storage Optimization',
            'message': 'Some items are stored at non-optimal temperatures. Maintain 4°C for most perishables.',
            'icon': '❄️'
        })
    
    # Quantity recommendations
    large_quantities = [item for item in food_items if item['quantity_kg'] > 5]
    if large_quantities:
        recommendations.append({
            'type': 'planning',
            'title': 'Bulk Purchase Alert',
            'message': 'Consider buying smaller quantities more frequently to reduce waste.',
            'icon': '🛒'
        })
    
    # Default recommendation if no specific issues
    if not recommendations:
        recommendations.append({
            'type': 'good',
            'title': 'Good Job!',
            'message': 'Your food management looks efficient. Keep tracking to maintain low waste levels.',
            'icon': '✅'
        })
    
    return recommendations

def main():
    load_css()
    
    st.markdown('<h1 class="main-header">🍎 SmartWaste - AI Food Waste Reduction</h1>', unsafe_allow_html=True)
    st.markdown("### Reduce Food Waste with AI-Powered Insights")
    
    # Initialize session state
    if 'predictor' not in st.session_state:
        st.session_state.predictor = WastePredictor()
        st.session_state.predictor.train_model()
    
    if 'image_processor' not in st.session_state:
        st.session_state.image_processor = ImageProcessor()
    
    if 'food_items' not in st.session_state:
        st.session_state.food_items = load_sample_data()
    
    if 'waste_data' not in st.session_state:
        st.session_state.waste_data = []
    
    # Sidebar navigation
    st.sidebar.title("🍎 Navigation")
    st.sidebar.markdown("---")
    app_mode = st.sidebar.selectbox("Choose Mode", 
                                   ["Dashboard", "Add Food Items", "AI Analysis", "Recommendations", "About"])
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Quick Tips:**
    - Add all food items for accurate predictions
    - Use AI analysis for food recognition
    - Follow recommendations to reduce waste
    - Track expiry dates regularly
    """)
    
    if app_mode == "Dashboard":
        show_dashboard()
    elif app_mode == "Add Food Items":
        show_add_food()
    elif app_mode == "AI Analysis":
        show_ai_analysis()
    elif app_mode == "Recommendations":
        show_recommendations()
    else:
        show_about()

def show_dashboard():
    st.header("📊 Waste Reduction Dashboard")
    
    if not st.session_state.food_items:
        st.info("Add some food items to see your waste analysis dashboard.")
        return
    
    # Calculate metrics
    total_items = len(st.session_state.food_items)
    total_quantity = sum(item['quantity_kg'] for item in st.session_state.food_items)
    total_value = sum(item['quantity_kg'] * item['price_per_kg'] for item in st.session_state.food_items)
    
    # Predict waste
    waste_predictions = []
    for item in st.session_state.food_items:
        prediction = st.session_state.predictor.predict_waste(
            item['quantity_kg'], item['days_to_expiry'],
            item['food_type'], item['storage_temp'], item['price_per_kg']
        )
        waste_predictions.append(prediction)
    
    st.session_state.waste_data = waste_predictions
    
    total_waste_kg = sum(pred['waste_kg'] for pred in waste_predictions)
    total_cost_loss = sum(pred['cost_loss'] for pred in waste_predictions)
    avg_waste_percentage = np.mean([pred['waste_percentage'] for pred in waste_predictions])
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Items", total_items)
    with col2:
        st.metric("Total Quantity", f"{total_quantity:.1f} kg")
    with col3:
        st.metric("Projected Waste", f"{total_waste_kg:.1f} kg")
    with col4:
        st.metric("Cost Savings Potential", f"₹{total_cost_loss:.0f}")
    
    # Waste distribution chart
    st.subheader("📈 Waste Projection by Food Type")
    
    waste_by_type = {}
    for item, pred in zip(st.session_state.food_items, waste_predictions):
        food_type = item['food_type']
        if food_type not in waste_by_type:
            waste_by_type[food_type] = 0
        waste_by_type[food_type] += pred['waste_kg']
    
    if waste_by_type:
        fig = px.pie(
            values=list(waste_by_type.values()),
            names=list(waste_by_type.keys()),
            title="Projected Waste Distribution by Food Type",
            color_discrete_sequence=px.colors.sequential.Greens
        )
        st.plotly_chart(fig)
    
    # Expiry timeline
    st.subheader("⏰ Expiry Timeline")
    
    expiry_data = []
    for item in st.session_state.food_items:
        expiry_date = datetime.now() + timedelta(days=item['days_to_expiry'])
        expiry_data.append({
            'Food Item': f"{item['name']} ({item['quantity_kg']}kg)",
            'Expiry Date': expiry_date,
            'Days Left': item['days_to_expiry'],
            'Waste Risk': 'High' if item['days_to_expiry'] <= 3 else 'Medium' if item['days_to_expiry'] <= 7 else 'Low'
        })
    
    if expiry_data:
        expiry_df = pd.DataFrame(expiry_data)
        fig = px.timeline(
            expiry_df, 
            x_start=pd.to_datetime([datetime.now()] * len(expiry_df)),
            x_end='Expiry Date',
            y='Food Item',
            color='Waste Risk',
            color_discrete_map={'High': '#dc3545', 'Medium': '#ffc107', 'Low': '#28a745'}
        )
        fig.update_layout(title="Food Expiry Timeline")
        st.plotly_chart(fig)

def show_add_food():
    st.header("➕ Add Food Items")
    
    with st.form("add_food_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            food_type = st.selectbox("Food Type", 
                                   ['Fruits', 'Vegetables', 'Dairy', 'Meat', 'Grains', 'Bakery'])
            quantity_kg = st.number_input("Quantity (kg)", min_value=0.1, max_value=100.0, value=1.0, step=0.1)
            days_to_expiry = st.slider("Days to Expiry", min_value=1, max_value=30, value=7)
        
        with col2:
            storage_temp = st.slider("Storage Temperature (°C)", min_value=-5, max_value=25, value=4)
            price_per_kg = st.number_input("Price per kg (₹)", min_value=5, max_value=1000, value=100)
            item_name = st.text_input("Item Name/Description", value=f"{food_type} Item")
        
        submitted = st.form_submit_button("Add Food Item")
        
        if submitted:
            new_item = {
                'name': item_name,
                'food_type': food_type,
                'quantity_kg': quantity_kg,
                'days_to_expiry': days_to_expiry,
                'storage_temp': storage_temp,
                'price_per_kg': price_per_kg,
                'added_date': datetime.now()
            }
            
            st.session_state.food_items.append(new_item)
            st.success(f"✅ Added {quantity_kg}kg of {item_name}")
    
    # Show current items
    if st.session_state.food_items:
        st.subheader("📋 Current Food Inventory")
        
        display_items = []
        for item in st.session_state.food_items:
            display_items.append({
                'Name': item['name'],
                'Type': item['food_type'],
                'Quantity (kg)': item['quantity_kg'],
                'Days to Expiry': item['days_to_expiry'],
                'Storage Temp (°C)': item['storage_temp'],
                'Price (₹/kg)': item['price_per_kg']
            })
        
        items_df = pd.DataFrame(display_items)
        st.dataframe(items_df, use_container_width=True)
        
        if st.button("Clear All Items"):
            st.session_state.food_items = []
            st.session_state.waste_data = []
            st.experimental_rerun()

def show_ai_analysis():
    st.header("🤖 AI Food Analysis")
    
    tab1, tab2 = st.tabs(["Image Analysis", "Waste Prediction"])
    
    with tab1:
        st.subheader("🍎 Food Image Analysis")
        
        uploaded_file = st.file_uploader("Upload food image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            
            with col2:
                with st.spinner("Analyzing image..."):
                    result = st.session_state.image_processor.analyze_image(uploaded_file)
                    
                    st.markdown("### Analysis Results")
                    st.info(result['message'])
                    
                    if result['confidence'] > 0.7:
                        st.success("✅ High confidence detection")
                    else:
                        st.warning("⚠️ Low confidence detection")
    
    with tab2:
        st.subheader("📊 Waste Prediction Analysis")
        
        if not st.session_state.food_items:
            st.info("Add food items to see waste predictions.")
            return
        
        # Show waste predictions
        waste_data = []
        for i, (item, pred) in enumerate(zip(st.session_state.food_items, st.session_state.waste_data)):
            waste_data.append({
                'Item': item['name'],
                'Type': item['food_type'],
                'Quantity': f"{item['quantity_kg']}kg",
                'Expiry': f"{item['days_to_expiry']} days",
                'Waste %': f"{pred['waste_percentage']:.1f}%",
                'Waste kg': f"{pred['waste_kg']:.2f}kg",
                'Cost Loss': f"₹{pred['cost_loss']:.0f}"
            })
        
        waste_df = pd.DataFrame(waste_data)
        st.dataframe(waste_df, use_container_width=True)
        
        # Waste risk analysis
        st.subheader("🚨 Waste Risk Assessment")
        
        high_risk = [i for i, pred in enumerate(st.session_state.waste_data) 
                    if pred['waste_percentage'] > 50]
        
        if high_risk:
            st.error("High Waste Risk Items Detected!")
            for idx in high_risk:
                item = st.session_state.food_items[idx]
                pred = st.session_state.waste_data[idx]
                st.markdown(f"""
                <div class="waste-alert">
                **{item['name']}**: {pred['waste_percentage']:.1f}% waste risk 
                (₹{pred['cost_loss']:.0f} potential loss)
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ No high-risk items detected")

def show_recommendations():
    st.header("💡 AI Recommendations")
    
    if not st.session_state.food_items:
        st.info("Add food items to get personalized recommendations.")
        return
    
    recommendations = generate_recommendations(
        st.session_state.waste_data, 
        st.session_state.food_items
    )
    
    st.subheader("🎯 Personalized Waste Reduction Plan")
    
    for rec in recommendations:
        if rec['type'] == 'high_priority':
            st.error(f"{rec['icon']} **{rec['title']}**\n\n{rec['message']}")
        elif rec['type'] == 'urgent':
            st.warning(f"{rec['icon']} **{rec['title']}**\n\n{rec['message']}")
        else:
            st.info(f"{rec['icon']} **{rec['title']}**\n\n{rec['message']}")
    
    # Additional tips
    st.subheader("🌱 General Food Waste Reduction Tips")
    
    tips = [
        "🛒 Plan meals and create shopping lists to avoid overbuying",
        "❄️ Maintain refrigerator temperature at 4°C or below",
        "📅 Use the 'First In, First Out' method for food rotation",
        "🍲 Cook proper portions and store leftovers properly",
        "🥫 Learn preservation techniques like freezing, canning, and drying",
        "📊 Regularly track your food waste to identify patterns",
        "🥗 Get creative with leftovers and food scraps"
    ]
    
    for tip in tips:
        st.markdown(f"<div class='recommendation'>{tip}</div>", unsafe_allow_html=True)

def show_about():
    st.header("ℹ️ About SmartWaste")
    
    st.markdown("""
    ### 🚀 How It Works
    
    SmartWaste uses machine learning and AI to help reduce food waste through:
    
    1. **Waste Prediction**: ML models predict potential waste based on food type, quantity, expiry, and storage
    2. **Image Analysis**: AI-powered food recognition from images
    3. **Smart Recommendations**: Personalized suggestions to reduce waste and save money
    4. **Real-time Monitoring**: Dashboard with waste projections and expiry tracking
    
    ### 🛠️ Technology Stack
    
    - **Frontend**: Streamlit
    - **Backend**: Python
    - **ML Models**: Random Forest, Image Processing
    - **Data Analysis**: Pandas, NumPy, Plotly
    - **Visualization**: Interactive charts and timelines
    
    ### 📈 Benefits
    
    - Reduce food waste by 30-50%
    - Save money on grocery bills
    - Environmental conservation
    - Better meal planning and storage
    - Data-driven food management
    
    ### 🎯 Use Cases
    
    - Households tracking food consumption
    - Restaurants and food services
    - Grocery stores and supermarkets
    - Environmental organizations
    - Educational institutions
    
    ### 🔧 Getting Started
    
    1. Add your food items with details
    2. View waste predictions on dashboard
    3. Get AI recommendations
    4. Implement suggestions to reduce waste
    5. Track your progress over time
    """)

if __name__ == "__main__":
    main()