import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import base64
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
import mysql.connector
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the model once at startup
model_best = load_model('food_scanner_model.keras')
food_list = ["Club Sandwich", "Cup Cakes", "Donuts", "Dumplings", "French Fries", "Ice Cream", "Omelette", "Pizza", "Samosa", "Spring Roll"]

# Load the CSV data at startup
csv_file_path = 'fooddex.csv'
df = pd.read_csv(csv_file_path)
df['Cals_per100grams'] = df['Cals_per100grams'].str.extract('(\d+)').astype(float)
df['FoodItem'] = df['FoodItem'].apply(lambda x: x.strip().lower())

def init_db_connection():
    try:
        conn = mysql.connector.connect(
            user=os.getenv('DB_USER', 'root'),
            password=os.getenv('DB_PASSWORD', 'mukul'),
            host=os.getenv('DB_HOST', 'localhost'),
            port=os.getenv('DB_PORT', 3306),
            database=os.getenv('DB_NAME', 'Food_scanner')
        )
        return conn
    except mysql.connector.Error as err:
        logger.error(f"Error connecting to MySQL: {err}")
        return None

def fetch_food_info(food_label, conn):
    try:
        cursor = conn.cursor()
        query = "SELECT * FROM food_info WHERE food_label=%s"
        cursor.execute(query, (food_label,))
        rows = cursor.fetchall()
        cursor.close()
        return rows
    except mysql.connector.Error as err:
        logger.error(f"Error fetching food info: {err}")
        return []

def fetch_all_foods(conn):
    try:
        cursor = conn.cursor()
        query = "SELECT food_label, unlocked FROM food_info"
        cursor.execute(query)
        rows = cursor.fetchall()
        cursor.close()
        return rows
    except mysql.connector.Error as err:
        logger.error(f"Error fetching all food items: {err}")
        return []

def update_food_status(food_label, conn):
    logger.info(f"Attempting to update food status for: {food_label}")
    try:
        # Normalize the food label to match the database format
        food_label = normalize_text(food_label)  
        
        cursor = conn.cursor()
        query = "UPDATE food_info SET unlocked = 1 WHERE food_label = %s"
        cursor.execute(query, (food_label,))
        conn.commit()
        
        # Check if the update affected any rows
        if cursor.rowcount == 0:
            logger.warning(f"No rows updated for food label: '{food_label}'. Check if the label exists and is correctly formatted.")
        else:
            logger.info(f"Successfully updated status for food label: '{food_label}'.")
        
        cursor.close()
        
    except mysql.connector.Error as err:
        logger.error(f"Error updating food status for '{food_label}': {err}")


def get_today_date():
    return pd.Timestamp('today').date()

def fetch_or_create_daily_calories(conn):
    today_date = get_today_date()

    try:
        cursor = conn.cursor()

        # Check if today's date exists
        query = "SELECT total_calories FROM daily_calories WHERE date = %s"
        cursor.execute(query, (today_date,))
        row = cursor.fetchone()

        if row:
            logger.info(f"Found existing entry for today ({today_date}), total calories: {row[0]}")
            return row[0]  # Return the current total calories
        else:
            # Insert a new row for today
            insert_query = "INSERT INTO daily_calories (date, total_calories) VALUES (%s, %s)"
            cursor.execute(insert_query, (today_date, 0.0))
            conn.commit()
            logger.info(f"Created new entry for today ({today_date})")
            return 0.0  # Initial calories for the new entry

    except mysql.connector.Error as err:
        logger.error(f"Error fetching or creating daily calories: {err}")
        return None

def update_daily_calories(conn, additional_calories):
    today_date = get_today_date()

    try:
        cursor = conn.cursor()

        # Update the total calories for today
        update_query = "UPDATE daily_calories SET total_calories = total_calories + %s WHERE date = %s"
        cursor.execute(update_query, (additional_calories, today_date))
        conn.commit()

        logger.info(f"Updated today's ({today_date}) total calories by adding {additional_calories} calories.")

    except mysql.connector.Error as err:
        logger.error(f"Error updating daily calories: {err}")

    finally:
        cursor.close()


@app.route('/')
def home():
    logger.info("Rendering home page...")
    return render_template('index.html')

@app.route('/loading')
def loading():
    return render_template('loading.html')


@app.route('/calorie_log')
def calorie_log():
    calorie_data = []
    try:
        conn = init_db_connection()
        if conn:
            with conn.cursor() as cursor:
                query = "SELECT * FROM daily_calories"
                cursor.execute(query)
                calorie_data = cursor.fetchall()
    except mysql.connector.Error as err:
        logger.error(f"Error fetching calorie data: {err}")
    finally:
        if conn:
            conn.close()

    return render_template('calorie_log.html', calorie_data=calorie_data)


@app.route('/fetch_calorie_data', methods=['GET'])
def fetch_calorie_data():
    dates = []
    calories = []
    try:
        conn = init_db_connection()
        if conn:
            with conn.cursor() as cursor:
                query = "SELECT date, total_calories FROM daily_calories ORDER BY date"
                cursor.execute(query)
                rows = cursor.fetchall()
                for row in rows:
                    dates.append(row[0])
                    calories.append(row[1])
    except mysql.connector.Error as err:
        logger.error(f"Error fetching calorie data for chart: {err}")
    finally:
        if conn:
            conn.close()

    return jsonify({'dates': dates, 'calories': calories})

@app.route('/unlocked')
def unlocked():
    conn = init_db_connection()
    food_items = fetch_all_foods(conn) if conn else []
    if conn:
        conn.close()
    return render_template('unlocked.html', food_items=food_items)

@app.route('/start_streamlit', methods=['POST'])
def start_streamlit_route():
    thread = threading.Thread(target=start_streamlit)
    thread.start()
    return jsonify({"message": "Streamlit application started"}), 200


@app.route('/process_upload', methods=['POST'])
def process_upload():
    try:
        data = json.loads(request.data)
        image_upload = data['imageUpload']
        weight_food = float(data.get('weightFood', 0.0))

        # Decode and save image
        image_data = image_upload.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.jpg')
        with open(image_path, 'wb') as f:
            f.write(image_bytes)

        # Process the image
        predicted_food_item = predict_food_item(image_path)

        # Database operations
        conn = init_db_connection()
        if conn:
            try:
                update_food_status(predicted_food_item, conn)
                # Fetch or create today's entry
                current_calories = fetch_or_create_daily_calories(conn)
                # Calculate the new calories
                new_calories = calculate_calories(predicted_food_item, weight_food, df)
                # Update today's entry with the new calories
                update_daily_calories(conn, new_calories)
                # Fetch food info for display
                rows = fetch_food_info(predicted_food_item, conn)
            finally:
                conn.close()
        else:
            rows = []
            new_calories = 0

        formatted_output = format_food_info(rows)
        encoded_image = generate_output_image(image_path, predicted_food_item, new_calories, weight_food)

        return jsonify({
            'image': f"data:image/png;base64,{encoded_image}",
            'output': formatted_output,
            'calories': new_calories
        })

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return jsonify({'error': 'An unexpected error occurred. Please try again later.'}), 500

def predict_food_item(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, (224, 224))
    image_normalized = image_resized.astype('float32') / 255.0
    image_batch = np.expand_dims(image_normalized, axis=0)

    predictions = model_best.predict(image_batch)
    predicted_index = np.argmax(predictions, axis=1)[0]
    return food_list[predicted_index]

def format_food_info(rows):
    if rows:
        row = rows[0]
        return f'''
        <div>
            <p><strong>ID:</strong> {row[0]}</p>
            <p><strong>Food Label:</strong> {row[1]}</p>
            <p><strong>Description:</strong> {row[2]}</p>
            <p><strong>Weight Range:</strong> {row[3]}</p>
            <p><strong>Calorie Range:</strong> {row[4]}</p>
            <p><strong>Country of Origin:</strong> {row[5]}</p>
            <p><strong>Common Ingredients:</strong> {row[6]}</p>
        </div>
        '''
    else:
        return "<p>No data found for the predicted food item.</p>"

def normalize_text(text):
    return text.strip().lower()

df['FoodItem'] = df['FoodItem'].apply(normalize_text)

import re

def clean_food_item(item):
    item = item.strip().lower()  # Basic normalization
    item = re.sub(r'\s+', ' ', item)  # Replace multiple spaces with a single space
    item = item.encode('ascii', 'ignore').decode('ascii')  # Remove non-ASCII characters
    return item

df['FoodItem'] = df['FoodItem'].apply(clean_food_item)


from fuzzywuzzy import process

def find_closest_food_item(query, food_list):
    closest_match, score = process.extractOne(query, food_list)
    return closest_match if score > 80 else None  # Adjust threshold as needed

def calculate_calories(food_item, weight, df):
    normalized_food_item = normalize_text(food_item)
    food_list = df['FoodItem'].tolist()
    closest_match = find_closest_food_item(normalized_food_item, food_list)
    
    if closest_match:
        cal_per_100g = df[df['FoodItem'] == closest_match]['Cals_per100grams'].values[0]
        total_calories = (cal_per_100g * weight) / 100
        return total_calories
    else:
        logger.warning(f"Food item '{food_item}' not found in the database.")
        return 0.0


def generate_output_image(image_path, predicted_food_item, calories, weight_food):
    image = cv2.imread(image_path)
    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.set_title(f"Predicted: {predicted_food_item}, Calories: {calories} cal for {weight_food} grams")
    ax.axis('off')

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return image_base64

if __name__ == '__main__':
    logger.info("Running the Flask app...")
    app.run(debug=True)
