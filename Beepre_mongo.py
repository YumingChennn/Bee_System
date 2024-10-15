import os
import io
import pymongo
import gridfs
import pickle
from scipy import stats
import numpy as np
import pandas as pd
import soundfile as sf
import librosa

# Step 1: Fetch audio files from MongoDB and process them
def fetch_audio_files_from_mongodb(limit=15):
    # Connect to MongoDB
    client = pymongo.MongoClient('mongodb://localhost:27017/')
    db = client['audio_db']  # Database with uploaded audio files and data
    fs = gridfs.GridFS(db)
    documents = db['caudio_db'].find().limit(limit)  # Assume data is in 'combined_data' collection
    audio_files = []
    audio_ids = []  # To keep track of the audio file IDs
    
    for doc in documents:
        audio_id = doc['audio_file_id']
        try:
            # Fetch audio file from GridFS using the file ID
            audio_data = fs.get(audio_id).read()
            audio_bytes = io.BytesIO(audio_data)  # Keep audio in memory
            
            # Read the audio file using soundfile (make sure audio is in a supported format)
            y, sr = sf.read(audio_bytes)
            audio_files.append((y, sr))
            audio_ids.append(audio_id)
        
        except Exception as e:
            print(f"Error processing audio ID {audio_id}: {e}")
            continue  # Skip to the next file in case of an error
        
    print(f"Total audio files fetched: {len(audio_files)}")
    return audio_files, audio_ids

# Audio chunk split
def split_audio_chunks(audio_files, chunk_size=5, hop_size=5):
    print("Splitting audio into chunks...")
    chunks = []

    for y, sr in audio_files:  # Here audio_files contains audio data and sample rate
        chunk_samples = int(chunk_size * sr)
        hop_samples = int(hop_size * sr)
        for start in range(0, len(y) - chunk_samples + 1, hop_samples):
            end = start + chunk_samples
            chunks.append(y[start:end])
    
    print(f"Total chunks created: {len(chunks)}")
    return chunks, sr  # Return sample rate

# Step 2: Load trained models
def load_models():
    with open('model/knn_model.pkl', 'rb') as file:
        knn_model = pickle.load(file)
    return knn_model

# Function to predict audio files
def predict_audio_files(audio_chunks, sample_rate, knn_model):
    all_knn_preds = []

    for chunk in audio_chunks:
        # Extract MFCC features
        mfcc_features = extract_mfcc_features(chunk, sample_rate)
        
        # KNN prediction
        knn_pred = knn_model.predict(mfcc_features.reshape(mfcc_features.shape[0], -1))
        all_knn_preds.append(knn_pred)

    # Calculate the final prediction using mode
    final_knn_pred = stats.mode(all_knn_preds, axis=0)[0][0]
    
    print(f'Final KNN Prediction: {final_knn_pred}')
    return final_knn_pred

# Function to extract MFCC features
def extract_mfcc_features(data, sample_rate, n_mfcc=20, chunk_size=22050):
    print("Extracting MFCC features...")
    mfcc_features = []
    
    num_chunks = len(data) // chunk_size
    for i in range(num_chunks):
        chunk = data[i*chunk_size:(i+1)*chunk_size]
        mfcc = librosa.feature.mfcc(y=chunk, sr=sample_rate, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_features.append(mfcc_mean)
    
    print(f"MFCC features extracted. Shape: {np.array(mfcc_features).shape}")
    return np.array(mfcc_features)

# Function to predict all audio files
def predict_all_audio_files(audio_files, audio_chunks, sample_rate, knn_model):
    print("Predicting all audio files...")
    predictions = []
    
    for i in range(len(audio_files)):
        prediction = predict_audio_files([audio_chunks[i]], sample_rate, knn_model)
        predictions.append(prediction)
    
    return predictions

def save_predictions_to_mongodb(audio_ids, predictions):
    # Connect to MongoDB
    client = pymongo.MongoClient('mongodb://localhost:27017/')
    db = client['audio_db']
    audio_collection = db['caudio_db']  # Collection to update predictions
    
    for audio_id, prediction in zip(audio_ids, predictions):
        # Convert numpy int64 to Python int
        prediction_value = int(prediction)  # Ensure it's a native Python int
        # Update the document with the new prediction
        audio_collection.update_one(
            {'audio_file_id': audio_id},  # Filter to find the document
            {'$set': {'prediction': prediction_value}}  # Add or update the prediction field
        )
    print(f"Updated predictions for {len(predictions)} audio files in MongoDB.")

# Main function
if __name__ == "__main__":
    print("Fetching audio files from MongoDB...")
    audio_files, audio_ids = fetch_audio_files_from_mongodb()
    
    print("Splitting audio into chunks...")
    audio_chunks, sample_rate = split_audio_chunks(audio_files)  # Get sample rate

    print("Loading models...")
    knn_model = load_models()
    
    print("Predicting audio files using loaded models...")
    predictions = predict_all_audio_files(audio_files, audio_chunks, sample_rate, knn_model)

    print("Saving predictions to MongoDB...")
    #save_predictions_to_mongodb(audio_ids, predictions)
