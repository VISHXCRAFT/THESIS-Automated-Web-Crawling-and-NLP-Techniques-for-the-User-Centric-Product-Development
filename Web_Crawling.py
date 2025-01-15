import requests
from bs4 import BeautifulSoup
import re
import time
import pandas as pd
from deep_translator import GoogleTranslator
import schedule
import json
import os

# List of known models
models_list = [
    'iMOW 5 EVO', 'iMOW 6 EVO', 'iMOW 7 EVO',
    'iMOW 5', 'iMOW 6', 'iMOW 7', 'iMOW 4',
    'iMOW 422', 'iMOW 522', 'iMOW 632'
]

# Regular expressions for model detection
model_patterns = {
    'iMOW 5 EVO': r'\bimow\s*5\s*evo\b',
    'iMOW 6 EVO': r'\bimow\s*6\s*evo\b',
    'iMOW 7 EVO': r'\bimow\s*7\s*evo\b|\bimow7evo\b|\bmower\s*7\s*evo\b|\bevo\s*7\b',
    'iMOW 4': r'\bimow\s*4\b(?!\s*evo)',
    'iMOW 5': r'\bimow\s*5\b(?!\s*evo)',
    'iMOW 6': r'\bimow\s*6\b(?!\s*evo)',
    'iMOW 7': r'\bimow\s*7\b(?!\s*evo)',
    'iMOW 422': r'\bimow\s*422\b',
    'iMOW 522': r'\bimow\s*522\b',
    'iMOW 632': r'\bimow\s*632\b',
}

combined_patterns = [
    r'\bimow\s*5\s*(,|/| and |&|\s)*6\s*(,|/| and |&|\s)*7\b',
    r'\bimow\s*5\s*-\s*7\b',
    r'\bimow\s*5-7\b',
]

def extract_models(text):
    found_models = set()
    text = text.lower()
    
    # Check for individual models
    for model, pattern in model_patterns.items():
        if re.search(pattern, text):
            found_models.add(model)
    
    # Check for combined models
    for pattern in combined_patterns:
        if re.search(pattern, text):
            found_models.update(['iMOW 4','iMOW 5', 'iMOW 6', 'iMOW 7', 'iMOW 5 EVO', 'iMOW 6 EVO', 'iMOW 7 EVO'])
            break
    
    return found_models

def categorize_model(title, first_post_content):
    # Check the title first
    title_models = extract_models(title)
    if title_models:
        if {'iMOW 4','iMOW 5', 'iMOW 6', 'iMOW 7', 'iMOW 5 EVO', 'iMOW 6 EVO', 'iMOW 7 EVO'}.issubset(title_models):
            return 'All'
        if 'evo' in title.lower() and any(model.endswith('EVO') for model in title_models):
            evo_model = next(model for model in title_models if model.endswith('EVO') or model.beginswith('EVO'))
            return evo_model
        return ', '.join(sorted(title_models))

    # Check the first post content if no models found in title
    content_models = extract_models(first_post_content)
    if content_models:
        if {'iMOW 5', 'iMOW 6', 'iMOW 7', 'iMOW 5 EVO', 'iMOW 6 EVO', 'iMOW 7 EVO'}.issubset(content_models):
            return 'All'
        return ', '.join(sorted(content_models))

    return 'General'

# Function to scrape forum data
def scrape_forum(url, crawled_urls):
    headers = {'User-Agent': 'Mozilla/5.0'}
    data = []
    model_counts = {}

    while url:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            threads = soup.find_all('div', class_='structItem-title')
            for thread in threads:
                thread_link = thread.find('a', href=True)
                if thread_link:
                    thread_url = 'https://www.roboter-forum.com' + thread_link['href']
                    thread_title = thread_link.text.strip()
                    if thread_url in crawled_urls:
                        continue
                    print(f"Processing thread: {thread_title} - {thread_url}")
                    first_post_content = get_first_post_content(thread_url, headers)
                    model = categorize_model(thread_title, first_post_content)
                    model_counts[model] = model_counts.get(model, 0) + 1

                    # Request each thread page
                    scrape_thread(thread_url, thread_title, model, data, headers)
                    crawled_urls.add(thread_url)
                    time.sleep(40)

            # Find the "Next" button for pagination
            next_button = soup.find('a', class_='pageNav-jump--next')
            url = 'https://www.roboter-forum.com' + next_button['href'] if next_button else None
        else:
            print(f"Failed to retrieve forum page with status code: {response.status_code}")
            return data, model_counts
    
    return data, model_counts

# Function to scrape each thread
def scrape_thread(thread_url, thread_title, model, data, headers):
    first_post_date = None
    while thread_url:
        thread_response = requests.get(thread_url, headers=headers)
        if thread_response.status_code == 200:
            thread_soup = BeautifulSoup(thread_response.text, 'html.parser')
            posts = thread_soup.find_all('article', class_='message')
            for post in posts:
                post_sequence = None
                post_sequence_element = post.find('ul', class_='message-attribution-opposite')
                if post_sequence_element:
                    post_sequence = post_sequence_element.find_all('li')[1].text.strip()

                post_author = post['data-author'] if 'data-author' in post.attrs else ''
                post_date = post.find('time')['datetime'] if post.find('time') else ''
                if not first_post_date:
                    first_post_date = post_date

                reactions_bar = post.find('div', class_='reactionsBar')
                if reactions_bar:
                    reactions_link = reactions_bar.find('a', class_='reactionsBar-link')
                    if reactions_link:
                        post_reactions_text = reactions_link.get_text(strip=True)
                        post_reactions_count = len(re.split(r',|und', post_reactions_text))
                    else:
                        post_reactions_count = 0
                else:
                    post_reactions_count = 0

                responses_count = len(post.find_all('blockquote')) if post.find_all('blockquote') else 0

                # Extract quotes and references
                response_to = []
                for quote in post.find_all('blockquote'):
                    response_to.append(quote['data-quote'] if 'data-quote' in quote.attrs else '')
                
                references = [a['href'] for a in post.find_all('a', href=True) if 'http' in a['href']]
                references_text = ', '.join(references)

                # Clean the main content
                post_content = post.find('div', class_='bbWrapper')
                quote_texts = extract_and_clean_quotes(post_content)
                bbCodeBlock_texts = extract_and_clean_bbCodeBlocks(post_content)

                post_content_cleaned = clean_main_content(post_content)

                if post_content_cleaned.strip():
                    # Translate post content
                    translated_post_content = translate_text(post_content_cleaned)
                    translated_thread_title = translate_text(thread_title)
                    translated_quote_texts = [translate_text(text) for text in quote_texts]
                    translated_bbCodeBlock_texts = [translate_text(text) for text in bbCodeBlock_texts]
                    
                    data.append({
                        'thread_title': translated_thread_title,
                        'post_content': translated_post_content,
                        'post_author': post_author,
                        'post_date': post_date,
                        'post_sequence': post_sequence,
                        'post_reactions': post_reactions_count,
                        'first_post_date': first_post_date,
                        'response_to': ', '.join(response_to),
                        'references': references_text,
                        'responses_count': responses_count,
                        'thread_url': thread_url,
                        'platform': 'roboter-forum.com',
                        'model': model,
                        'quote_text': ' | '.join(translated_quote_texts),
                        'bbCodeBlock_text': ' | '.join(translated_bbCodeBlock_texts)
                    })
                    print(f"Scraped data: {data[-1]}")
            
            next_button = thread_soup.find('a', class_='pageNav-jump--next')
            thread_url = 'https://www.roboter-forum.com' + next_button['href'] if next_button else None
        else:
            print(f"Failed to retrieve thread page with status code: {thread_response.status_code}")
            break

def get_first_post_content(thread_url, headers):
    response = requests.get(thread_url, headers=headers)
    if response.status_code == 200:
        thread_soup = BeautifulSoup(response.text, 'html.parser')
        first_post = thread_soup.find('article', class_='message')
        if first_post:
            content = first_post.find('div', class_='bbWrapper')
            return content.get_text(' ', strip=True) if content else ''
    return ''

def clean_main_content(bbWrapper):
    for quote in bbWrapper.find_all('blockquote'):
        quote.decompose()
    for link in bbWrapper.find_all('a', href=True):
        link.decompose()
    for bb_code in bbWrapper.find_all('div', class_='bbCodeBlock'):
        bb_code.decompose()
    return bbWrapper.get_text(' ', strip=True)

def extract_and_clean_quotes(bbWrapper):
    quote_texts = []
    for quote in bbWrapper.find_all('blockquote'):
        quote_texts.append(quote.get_text(' ', strip=True))
        quote.decompose()
    return quote_texts

def extract_and_clean_bbCodeBlocks(bbWrapper):
    bbCodeBlock_texts = []
    for bb_code in bbWrapper.find_all('div', class_='bbCodeBlock'):
        bbCodeBlock_texts.append(bb_code.get_text(' ', strip=True))
        bb_code.decompose()
    return bbCodeBlock_texts

def translate_text(text):
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception as e:
        print(f"Translation error: {e}")
        return text

def save_to_excel(data, model_counts, file_name='Web Crawled.xlsx', model_file_name='modelCrawled.xlsx'):
    data_saved = False
    model_data_saved = False
    rows_added = 0

    while not data_saved:
        try:
            if os.path.exists(file_name):
                existing_data = pd.read_excel(file_name)
                df = pd.DataFrame(data)
                rows_added = len(df)
                df = pd.concat([existing_data, df], ignore_index=True)
            else:
                df = pd.DataFrame(data)
                rows_added = len(df)
            df.to_excel(file_name, index=False)
            data_saved = True
        except PermissionError:
            print(f"Unable to save data to {file_name} because the file is open. Retrying in 20 seconds...")
            time.sleep(20)

    while not model_data_saved:
        try:
            if os.path.exists(model_file_name):
                existing_model_counts = pd.read_excel(model_file_name, index_col=0).to_dict()
                for key, value in existing_model_counts.items():
                    if key in model_counts:
                        model_counts[key] += value
            pd.Series(model_counts).to_excel(model_file_name)
            model_data_saved = True
        except PermissionError:
            print(f"Unable to save model data to {model_file_name} because the file is open. Retrying in 20 seconds...")
            time.sleep(20)
    
    print(f"Added {rows_added} rows to {file_name}")

def load_crawled_urls(file_name='crawled_urls.json'):
    if os.path.exists(file_name):
        with open(file_name, 'r') as f:
            return set(json.load(f))
    return set()

def save_crawled_urls(crawled_urls, file_name='crawled_urls.json'):
    with open(file_name, 'w') as f:
        json.dump(list(crawled_urls), f)

def main():
    forum_url = 'https://www.roboter-forum.com/forums/imow-5-6-7-evo.255/'
    crawled_urls = load_crawled_urls()
    data, model_counts = scrape_forum(forum_url, crawled_urls)

    if not data:
        print("No New Comments for now. Check in 2 hours")
    else:
        save_to_excel(data, model_counts)
        save_crawled_urls(crawled_urls)
        print("Data extraction complete. Check the generated Excel files.")

# Schedule the script to run every 2 hours
schedule.every(2).hours.do(main)

if __name__ == '__main__':
    main()
    while True:
        schedule.run_pending()
        time.sleep(1)
