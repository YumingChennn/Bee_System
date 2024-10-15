from flask import Flask, jsonify
from pymongo import MongoClient
from flask_cors import CORS
from bson import ObjectId
import datetime

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://127.0.0.1:5500"}})

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['audio_db']
collection = db['caudio_db']

# Function to convert non-serializable objects
def json_converter(data):
    for item in data:
        if 'Date' in item:
            item['Date'] = item['Date'].strftime('%Y-%m-%d %H:%M:%S')  # Convert datetime to string
        if 'audio_file_id' in item:
            item['audio_file_id'] = str(item['audio_file_id'])  # Convert ObjectId to string
    return data

@app.route('/data', methods=['GET'])
def get_data():
    try:
        # Fetch all documents from MongoDB
        data = list(collection.find({}, {'_id': 0}))  # Exclude '_id' field
        # Convert datetime and ObjectId to JSON serializable format
        data = json_converter(data)

        # Calculate the number of 0's in the 'prediction' field
        zero_predictions_count = sum(item.get('prediction', None) == 0 for item in data)

        # Include the count in the response
        return jsonify({
            'data': data,
            'zero_predictions_count': zero_predictions_count
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
