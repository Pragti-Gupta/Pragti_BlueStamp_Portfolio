import google.generativeai as genai
from PIL import Image
import requests
from picamera2 import Picamera2
import cv2
import time



picam2 = Picamera2()
picam2.start()

# Trigger autofocus
picam2.set_controls({"AfMode": 2})
time.sleep(2)  # time for autofocusing to work

# Capture focused image
image_path = "autofocused_image2.jpg"
picam2.capture_file(image_path)
picam2.stop()


# Setup
GOOGLE_API_KEY = "YOUR_API_KEY"
genai.configure(api_key=GOOGLE_API_KEY)

# Load image
def load_image_as_base64(path):
    with open(path, "rb") as img_file:
        return img_file.read()

image_path = "autofocused_image2.jpg"
image_bytes = load_image_as_base64(image_path)

# Create Gemini model instance
model = genai.GenerativeModel("gemini-1.5-flash")

# Make OCR-like prompt
response = model.generate_content([
    "Extract all readable text from this image (OCR):",
    {"mime_type": "image/png", "data": image_bytes}
])





def search_google_books(query, max_results=5):
    url = "https://www.googleapis.com/books/v1/volumes"
    params = {
        "q": query,
        "maxResults": max_results,
        "key": "YOUR_API_KEY", 
    }

    response = requests.get(url, params=params)
    data = response.json()

    if "items" not in data:
        print("No results found.")
        return

    for i, item in enumerate(data["items"], 1):
        volume_info = item.get("volumeInfo", {})
        title = volume_info.get("title", "N/A")
        authors = volume_info.get("authors", ["N/A"])
        publisher = volume_info.get("publisher", "N/A")
        published_date = volume_info.get("publishedDate", "N/A")
        rating = volume_info.get("averageRating", "No rating")
        ratings_count = volume_info.get("ratingsCount", 0)
        description = volume_info.get("description", "No description.")
        page_count = volume_info.get("pageCount", "Unknown")
        

        print(f"\nResult {i}:")
        print(f"Title: {title}")
        print(f"Author(s): {', '.join(authors)}")
        print(f"Publisher: {publisher}")
        print(f"Published: {published_date}")
        print(f"Description: {description}")
        print(f"Rating: {rating} ({ratings_count} ratings)")  
        print(f"Page Count: {page_count}")  


print("Extracted Text:\n", response.text)
search_google_books(response.text)



   
recommendations = model.generate_content([
    "Get 5 recommendations similar to this book title: "+ response.text
])

series = model.generate_content([
    "Tell me if this book title: "+ response.text + "is part of a book series. If so tell me the other books. Don't start off with yes or no"+
    "can you say the number of books in the series first"
])
print(recommendations.text)
print(series.text)
