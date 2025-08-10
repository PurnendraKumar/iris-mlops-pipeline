import logging
import sqlite3
import json
from datetime import datetime
import os

class PredictionLogger:
    def __init__(self, db_path='logs/predictions.db'):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_db()
    
    def init_db(self):
        """Initialize SQLite database for predictions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                input_data TEXT NOT NULL,
                prediction INTEGER NOT NULL,
                prediction_label TEXT NOT NULL,
                confidence REAL NOT NULL
            )
        ''')
        conn.commit()
        conn.close()
    
    def log_prediction(self, input_data, prediction, prediction_label, confidence):
        """Log prediction to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions 
            (timestamp, input_data, prediction, prediction_label, confidence)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            json.dumps(input_data),
            prediction,
            prediction_label,
            confidence
        ))
        conn.commit()
        conn.close()
    
    def get_prediction_stats(self):
        """Get prediction statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total predictions
        cursor.execute('SELECT COUNT(*) FROM predictions')
        total = cursor.fetchone()[0]
        
        # Predictions by class
        cursor.execute('''
            SELECT prediction_label, COUNT(*) 
            FROM predictions 
            GROUP BY prediction_label
        ''')
        by_class = dict(cursor.fetchall())
        
        # Average confidence
        cursor.execute('SELECT AVG(confidence) FROM predictions')
        avg_confidence = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            'total_predictions': total,
            'predictions_by_class': by_class,
            'average_confidence': round(avg_confidence, 4)
        }