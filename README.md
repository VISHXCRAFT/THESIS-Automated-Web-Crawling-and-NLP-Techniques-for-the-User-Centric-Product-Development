# THESIS-Automated-Web-Crawling-and-NLP-Techniques-for-the-User-Centric-Product-Development
Repository Description for Thesis:
Title: Automated Web Crawling and NLP Techniques for the User-Centered Product Development

![Dashboard Overview]![image](https://github.com/user-attachments/assets/bb1359bf-2187-4e3d-b7f2-8c445ed543ee)


**Overview:**
This repository contains the code and resources related to my Master's thesis, which focuses on automated web crawling and the application of natural language processing (NLP) techniques to gather and analyze user-generated data from social media platforms. The primary goal of this research was to derive actionable insights that can inform product development, marketing strategies, and customer engagement for user-centered businesses.

**Key Features:**
1)  Web Crawling:
      The script automates the process of collecting unstructured data (e.g., user reviews, feedback, and posts) from various social media       platforms such as forums or review sites.
      Uses **requests** and **BeautifulSoup** for scraping web pages and handling raw HTML data.

2)  Data Categorization and Analysis:
      Extracted comments and feedback are categorized based on various factors like sentiment, emotion, technical issues, feature                requests, and questions.
      The script uses a **Zero-Shot Classification** model to automatically classify comments into predefined categories
      (e.g.,"connectivity", "installation", "feature request").
      **Emotion Analysis** is performed on each comment using a pre-trained emotion detection model, classifying emotions like                   "frustration", "satisfaction", or "anticipation".
      Sentiment analysis is done using **TextBlob**, assigning a sentiment label (positive, negative, neutral) based on the content.

3)  Post-processing and Categorization:
      Comments are analyzed to detect keywords related to various technical categories such as blades, hardware, connectivity, software,         and installation.
      Specific iMOW model mentions are detected, and relevant countries or regions mentioned are extracted.
      The categorization results are organized into a structured format with information about the sentiment, emotions, technical issues,        feature requests, and more.

4)  Data Export:
      The processed data is exported into an Excel file, with the results of the analysis clearly laid out.
      The script supports incremental processing, meaning only new or unprocessed rows are analyzed, making it suitable for ongoing              monitoring.

5)  Automated Execution:
      The script is set to run continuously in the background, checking for new data every 2 hours, and appending the categorized results        to the output file.

      ![Screenshot 2025-02-27 105222](https://github.com/user-attachments/assets/7634cc70-ee3c-4c22-92f1-25563da2e873)

**Technologies Used:**

Python: The core language for the script.

Libraries:
**Pandas**: For data manipulation and Excel file handling.
**TextBlob**: For sentiment analysis.
**Transformers**: For zero-shot classification and emotion analysis using pre-trained models.
**Regular Expressions**: For detecting keywords and patterns in text data.
**Concurrent Futures**: For running the categorization in parallel, improving performance.
**Logging**: For monitoring and error reporting.
