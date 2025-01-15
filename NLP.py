import pandas as pd
import re
from textblob import TextBlob
from transformers import pipeline
import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load pre-trained models once
classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli', revision='main')
emotion_pipeline = pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base', top_k=None)

# Define categories for zero-shot classification
candidate_labels = ["connectivity", "installation", "other technical issue", "feature request"]

# Keywords for specific technical categories
technical_keywords = {
    "blades": r"\b(?:blade|blades|knive|knives)\b",
    "hardware": r"\b(?:dock|docking station|spare parts|wheel|wheels|tire|tyre|rotation|motor)\b",
    "connectivity": r"\b(?:server|mobile network|WLAN|GPS|Bluetooth|connectivity)\b",
    "software": r"\b(?:software version|standard settings|driving functions|functions)\b",
    "installation": r"\b(?:mowing time|install|installation|guide wire|boundary wire)\b",
    "design": r"\b(?:deck|design)\b",
    "app": r"\b(?:app|application|mobile app|software app|iMOW app)\b"
}

# Countries
countries = ["Germany", "UK", "Russia", "China", "India", "Italy", "Brazil", "USA", "Czech Republic", "Austria"]

# iMOW Models
imow_models = {
    'iMOW 7 EVO': r'\bstihl\s*imow\s*7\s*evo\b|\bstihl\s*imow7evo\b|\bstihl\s*mower\s*7\s*evo\b|\bevo\s*7\b',
    'iMOW 7': r'\bstihl\s*imow\s*7\b|\bstihl\s*imow7\b|\bstihl\s*mower\s*7\b|\biMOW\s*7\b',
    'iMOW 6 EVO': r'\bstihl\s*imow\s*6\s*evo\b|\bstihl\s*imow6evo\b|\bstihl\s*mower\s*6\s*evo\b|\bevo\s*6\b',
    'iMOW 6': r'\bstihl\s*imow\s*6\b|\bstihl\s*imow6\b|\bstihl\s*mower\s*6\b|\biMOW\s*6\b',
    'iMOW 5 EVO': r'\bstihl\s*imow\s*5\s*evo\b|\bstihl\s*imow5evo\b|\bstihl\s*mower\s*5\s*evo\b|\bevo\s*5\b',
    'iMOW 5': r'\bstihl\s*imow\s*5\b|\bstihl\s*imow5\b|\bstihl\s*mower\s*5\b|\biMOW\s*5\b',
    'iMOW 4': r'\bstihl\s*imow\s*4\b|\bstihl\s*imow4\b|\bstihl\s*mower\s*4\b|\biMOW\s*4\b',
    'iMOW 522': r'\bstihl\s*imow\s*522\b|\bstihl\s*mower\s*522\b|\biMOW\s*522\b|\biMOW\s*522C\b|\biMOW\s*522PC\b',
    'iMOW 422': r'\bstihl\s*imow\s*422\b|\bstihl\s*mower\s*422\b|\biMOW\s*422\b|\biMOW\s*422C\b|\biMOW\s*422PC\b',
    'iMOW 632': r'\bstihl\s*imow\s*632\b|\bstihl\s*mower\s*632\b|\biMOW\s*632\b|\biMOW\s*632C\b|\biMOW\s*632PC\b',
    'STIHL automower': r'\bstihl\s*automower\b',
    'STIHL robot': r'\bstihl\s*robot\b',
    'STIHL mower': r'\bstihl\s*mower\b',
    'STIHL robot mower': r'\bstihl\s*robot\s*mower\b',
    'STIHL iMOW': r'\bstihl\s*imow\b',
    'STIHL mower robot': r'\bstihl\s*mower\s*robot\b',
    'STIHL robotic mower': r'\bstihl\s*robotic\s*mower\b'
}

# Emotion keywords
emotion_keywords = {
    "anticipation": ["curiosity", "excited", "positive attitude", "hopeful", "possibility", "urgency", "forward thinking"],
    "frustration": ["frustration", "errors", "system", "angry", "disgusted", "disappointment"],
    "inquiry": ["solution", "request", "information", "question", "wondering", "gratitude"],
    "preference": ["desired outcome", "suggestions", "feature request", "improvement", "replacement", "old models", "old software", "old technology"],
    "suggestive": ["sharing information", "sharing tools", "sharing experience", "suggestions", "helpful"],
    "satisfied": ["happy", "joy", "joyful", "excited"]
}

# Function to detect keywords in a text and return matches
def detect_keywords_with_matches(text, patterns):
    detected = []
    matches = []
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            detected.append(key)
            matches.append(match.group(0))
    return detected, matches

# Function to detect keywords in a text
def detect_keywords(text, patterns):
    detected = []
    for key, pattern in patterns.items():
        if re.search(pattern, text, re.IGNORECASE):
            detected.append(key)
    return detected

# Function to categorize comments
def categorize_comments(data):
    categorized_data = []
    batch_size = 8  # Adjust batch size based on your system's capabilities

    total_items = len(data)
    processed_items = 0

    for i in range(0, total_items, batch_size):
        batch = data[i:i+batch_size]

        post_contents = [str(item['post_content']) if pd.notna(item['post_content']) else "" for item in batch]

        # Perform sentiment analysis using TextBlob
        sentiment_results = [TextBlob(content).sentiment.polarity for content in post_contents]

        # Perform zero-shot classification in batch
        try:
            classification_results = classifier(post_contents, candidate_labels)
        except Exception as e:
            logging.error(f"Error during classification: {e}")
            classification_results = []

        # Perform emotion classification in batch
        try:
            emotion_results = emotion_pipeline(post_contents)
        except Exception as e:
            logging.error(f"Error during emotion classification: {e}")
            emotion_results = []

        for j, item in enumerate(batch):
            post_content = item['post_content'].lower() if pd.notna(item['post_content']) else ""

            # Sentiment analysis
            sentiment_label = "neutral"
            if j < len(sentiment_results):
                sentiment = sentiment_results[j]
                if sentiment < -0.1:
                    sentiment_label = "negative"
                elif sentiment > 0.1:
                    sentiment_label = "positive"

            # Zero-shot classification
            technical_issues = []
            feature_request = "no"

            if j < len(classification_results):
                try:
                    scores = classification_results[j]['scores']
                    labels = classification_results[j]['labels']

                    for score, label in zip(scores, labels):
                        if score > 0.3:  # Adjust threshold for feature request detection
                            if label == "feature request":
                                feature_request = "yes"
                            else:
                                technical_issues.append(label)
                except Exception as e:
                    logging.error(f"Error processing classification result: {e}")

            # Manual keyword matching for technical categories
            detected_technical_issues, keyword_matches = detect_keywords_with_matches(post_content, technical_keywords)
            if detected_technical_issues:
                technical_issues.extend(detected_technical_issues)

            # Remove "app" if other more specific categories are detected
            if "app" in technical_issues and any(issue in technical_issues for issue in ["blades", "hardware", "connectivity", "software", "installation", "design"]):
                technical_issues.remove("app")

            # Ensure correct classification for connectivity issues
            if "connectivity" in technical_issues and any(word in post_content for word in ["serial number", "PM"]):
                technical_issues.remove("connectivity")

            # Emotion detection
            dominant_emotion = "neutral"
            dominant_emotion_score = 0.0
            if j < len(emotion_results):
                try:
                    emotion_scores = emotion_results[j]
                    if emotion_scores:
                        # Filter and sort emotions by score
                        filtered_emotions = [(e['label'], e['score']) for e in emotion_scores if e['label'] in emotion_keywords]
                        if filtered_emotions:
                            dominant_emotion, dominant_emotion_score = max(filtered_emotions, key=lambda x: x[1])
                except Exception as e:
                    logging.error(f"Error processing emotion result: {e}")

            # Manual keyword matching for emotion analysis
            for emotion, keywords in emotion_keywords.items():
                if any(keyword in post_content for keyword in keywords):
                    dominant_emotion = emotion
                    break

            # Detect countries
            detected_countries = [country for country in countries if country.lower() in post_content]

            # Detect iMOW models
            detected_imow_models = detect_keywords(post_content, imow_models)

            # Detect questions
            questions = []
            if any(word in post_content for word in ["what", "why", "how", "where", "when", "which"]):
                if re.search(technical_keywords["hardware"], post_content):
                    questions.append("hardware")
                elif re.search(technical_keywords["connectivity"], post_content):
                    questions.append("connectivity")
                elif re.search(technical_keywords["installation"], post_content):
                    questions.append("installation")
                elif re.search(technical_keywords["software"], post_content):
                    questions.append("software")
                else:
                    questions.append("general")

            # Detect solutions
            solutions = []
            if any(word in post_content for word in ["solution", "resolved", "fix", "answer"]) or item.get('reference'):
                solutions.append("solution")

            # Detect replacements
            replacements = []
            if any(word in post_content for word in ["updated", "changed", "replaced", "alternative"]):
                replacements.append("replaced")

            # Detect comparisons
            comparisons = []
            if any(word in post_content for word in ["comparison", "updated", "alternative"]):
                comparisons.append("comparison")

            # Detect complaints
            complaints = []
            if any(word in post_content for word in ["angry", "issue", "problem", "complaint"]):
                complaints.append("complaint")

            # Append categorized data with keyword matches for technical issues
            categorized_data.append({
                'thread_title': item['thread_title'],
                'post_author': item['post_author'],
                'thread_url': item['thread_url'],
                'post_content': item['post_content'],
                'sentiment': sentiment_label,
                'emotions': dominant_emotion,
                'emotion_score': dominant_emotion_score,
                'questions': ', '.join(set(questions)),
                'technical_issues': ', '.join(set(technical_issues)),
                'technical_issue_keywords': ', '.join(set(keyword_matches)),
                'country': ', '.join(set(detected_countries)),
                'iMOW_models': ', '.join(set(detected_imow_models)),
                'solution': ', '.join(set(solutions)),
                'replaced': ', '.join(set(replacements)),
                'feature_request': feature_request,
                'comparison': ', '.join(set(comparisons)),
                'complaints': ', '.join(set(complaints))
            })

            processed_items += 1
            logging.info(f"Processed {processed_items}/{total_items} items")

    return categorized_data

def process_new_data(excel_file, output_file, last_processed_index):
    try:
        # Load the crawled data from the existing Excel file
        data = pd.read_excel(excel_file)

        if 'id' in data.columns:
            # Get the new rows that have not been processed yet
            new_data = data[data['id'] > last_processed_index]  # Assuming 'id' is a unique identifier
        else:
            # Use the row index as a unique identifier
            data.reset_index(drop=True, inplace=True)
            new_data = data.iloc[last_processed_index:]

        if new_data.empty:
            logging.info("No new data to process.")
            return last_processed_index

        # Process comments to categorize them
        categorized_data = categorize_comments(new_data.to_dict('records'))

        # Convert categorized data to DataFrame
        categorized_df = pd.DataFrame(categorized_data)

        if os.path.exists(output_file):
            # Load existing output file
            existing_df = pd.read_excel(output_file)
            # Append new data
            final_df = pd.concat([existing_df, categorized_df], ignore_index=True)
        else:
            # Create a new output file
            final_df = categorized_df

        # Save the updated DataFrame to a new Excel file
        final_df.to_excel(output_file, index=False)

        # Update last processed index
        last_processed_index = data.index.max() + 1

        logging.info("NLP processing complete. Check the output file.")

    except PermissionError:
        logging.error("Error: Excel file is open. Please close the file and try again.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

    return last_processed_index

def main():
    excel_file = "C:\\Users\\Vishwas Goswami\\OneDrive\\Documents\\Final Thesis\\Web Crawled.xlsx"
    output_file = 'NLP Analysis.xlsx'
    last_processed_index = 0  # Initialize this with the index of the last processed row

    while True:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(process_new_data, excel_file, output_file, last_processed_index)
            last_processed_index = future.result()
        
        time.sleep(2 * 60 * 60)  # Wait for 2 hours

if __name__ == "__main__":
    main()
