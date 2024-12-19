import os
import requests
from bs4 import BeautifulSoup
import time

# Base URLs
GUTENBERG_BASE_URL = "https://www.gutenberg.org"
SHAKESPEARE_COLLECTION_URL = "https://www.gutenberg.org/ebooks/author/65"

# Directory to save the text files
OUTPUT_DIR = "shakespeare_works"

def fetch_page(url):
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch {url}: {response.status_code}")
        return None
    return BeautifulSoup(response.text, "html.parser")

def clean_gutenberg_text(text):
    lines = text.splitlines()
    start_index = None
    end_index = None

    # Find start marker
    for i, line in enumerate(lines):
        if "*** START OF THIS PROJECT GUTENBERG EBOOK" in line.upper():
            start_index = i
            break

    # Find end marker
    for i, line in enumerate(lines):
        if "*** END OF THIS PROJECT GUTENBERG EBOOK" in line.upper():
            end_index = i
            break

    # If both markers found, extract only the text in between
    if start_index is not None and end_index is not None and end_index > start_index:
        # The actual text usually starts after the start marker line
        lines = lines[start_index+1:end_index]
    else:
        # If markers not found, text remains as is.
        pass

    cleaned_text = "\n".join(lines)
    return cleaned_text

def download_work(work_url, title):
    response = requests.get(work_url)
    if response.status_code != 200:
        print(f"Failed to download {work_url}: {response.status_code}")
        return

    raw_text = response.text
    cleaned_text = clean_gutenberg_text(raw_text)

    # Save the cleaned text
    filename = f"{OUTPUT_DIR}/{title}.txt"
    with open(filename, "w", encoding="utf-8") as file:
        file.write(cleaned_text)
    print(f"Saved: {filename}")

def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Fetch Shakespeare's collection page
    soup = fetch_page(SHAKESPEARE_COLLECTION_URL)
    if not soup:
        print("Failed to load the Shakespeare collection page.")
        return

    # Find all book links on the page
    book_links = soup.select('li.booklink a[href]')
    if not book_links:
        print("No book links found.")
        return

    for link in book_links:
        book_title = link.get('title', 'Unknown Title').replace("/", "_")  # Handle special characters
        book_url = GUTENBERG_BASE_URL + link['href']
        print(f"Processing: {book_title} - {book_url}")

        # Navigate to the book's page to find the plain text download link
        book_soup = fetch_page(book_url)
        if not book_soup:
            continue

        # Look for plain text link (UTF-8 version preferred)
        plain_text_link = book_soup.find("a", string="Plain Text UTF-8")
        if not plain_text_link:
            print(f"No plain text link found for: {book_title}")
            continue

        plain_text_url = GUTENBERG_BASE_URL + plain_text_link['href']
        print(f"Downloading: {book_title} from {plain_text_url}")
        download_work(plain_text_url, book_title)

        # Delay to prevent overloading the server
        time.sleep(2)

if __name__ == "__main__":
    main()

import os

OUTPUT_DIR = "shakespeare_works"
combined_text_file = "data_shakespeare.txt"

combined_text = ""
for filename in os.listdir(OUTPUT_DIR):
    with open(os.path.join(OUTPUT_DIR, filename), "r", encoding="utf-8") as file:
        combined_text += file.read().strip() + "\n\n"

with open(combined_text_file, "w", encoding="utf-8") as file:
    file.write(combined_text)
print(f"Combined texts saved to {combined_text_file}")