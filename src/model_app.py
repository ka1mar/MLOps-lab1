
from flask import Flask, request, jsonify
import logging
import pandas as pd
from catboost import CatBoostClassifier, Pool
import configparser

class ModelManager:
    """Class to manage ML model loading and inference"""
    
    def __init__(self, config_path='config.ini'):
        """Initialize the model manager with configuration"""
        # Load config
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        
        # Get model path from config
        self.model_path = self.config["MODEL"]["path"]
        
        # Setup logger
        self.logger = logging.getLogger('ModelManager')
        
        # Load model
        self.model = self._load_model()
    
    def _load_model(self):
        """Load the machine learning model"""
        model = CatBoostClassifier()
        print('\n\n\n\n')
        print("self.model_path", self.model_path)
        model.load_model(self.model_path)
        self.logger.info(f"Model loaded from {self.model_path}")
        return model
    
    def predict(self, features):
        """Make predictions using the loaded model"""
        # Convert input to DataFrame
        input_data = pd.DataFrame(features)
        
        # Create test pool and make predictions
        test_pool = Pool(input_data)
        predictions = self.model.predict(test_pool)
        
        return predictions.tolist()


class PredictionService:
    """Service class to handle prediction requests"""
    
    def __init__(self, model_manager):
        """Initialize with model manager"""
        self.model_manager = model_manager
        self.logger = logging.getLogger('PredictionService')
    
    def process_request(self, request_data):
        """Process the prediction request"""
        if not request_data.is_json:
            return {"error": "Request must be JSON"}, 400
            
        data = request_data.get_json()
        
        if "features" not in data:
            return {"error": "Request JSON must contain 'features' key"}, 400
            
        try:
            predictions = self.model_manager.predict(data["features"])
            response = {"predictions": predictions}
            return response, 200
        except Exception as e:
            self.logger.error(f"Error in prediction: {e}")
            return {"error": str(e)}, 500


class FlaskApp:
    """Flask application wrapper"""
    
    def __init__(self):
        """Initialize the Flask application"""
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
        # Create Flask app
        self.app = Flask(__name__)
        
        # Initialize model manager
        self.model_manager = ModelManager()
        
        # Initialize prediction service
        self.prediction_service = PredictionService(self.model_manager)
        
        # Register routes
        self._register_routes()
    
    def _register_routes(self):
        """Register API routes"""
        self.app.route('/predict', methods=['POST'])(self.predict)
    
    def predict(self):
        """Handle prediction requests"""
        response, status_code = self.prediction_service.process_request(request)
        return jsonify(response), status_code
    
    def run(self, host='0.0.0.0', port=5000):
        """Run the Flask application"""
        self.app.run(host=host, port=port)


# Main application entry point
if __name__ == '__main__':
    app = FlaskApp()
    app.run()
