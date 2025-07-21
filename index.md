# Smart Glasses
The smart glasses help you recognize the objects around you in the world. Using rapberry pi and a picam, I am able to detect objects using tensor flow, which also has a speech output. So, whenever an object is put in front of it, it is able to detect the object and say the name of the object detected outloud.<!-- need to add more-->

| **Engineer** | **School** | **Area of Interest** | **Grade** |
|:--:|:--:|:--:|:--:|
| Pragti G | Monta Vista High School | Software Engineering | Incoming Junior

<!--Replace the BlueStamp logo below with an image of yourself and your completed project. Follow the guide [here](https://tomcam.github.io/least-github-pages/adding-images-github-pages-site.html) if you need help. -->
<img src = "PragtiG.png" style = "width:30%; height:30%;">
  
# Final Milestone

**Don't forget to replace the text below with the embedding for your milestone video. Go to Youtube, click Share -> Embed, and copy and paste the code to replace what's below.**

<iframe width="560" height="315" src="https://www.youtube.com/embed/H6bi_PE4X9o?si=RHki3qWzc1t0U80w" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

For your final milestone, explain the outcome of your project. Key details to include are:
- What you've accomplished since your previous milestone
- What your biggest challenges and triumphs were at BSE
- A summary of key topics you learned about
- What you hope to learn in the future after everything you've learned at BSE

## Summary
I created a website with streamlit to take a picture of a book with your device camera, detect the text using a gemini api, provide details with a google books api, and get reccomendations from my own algorithm.

## Steps
I had gotten a new camera with better quality, hoping it would improve OCR—and it probably would—but when I put the filters, it would detect every small texture as a word or character, messing up the detection. So, I decided to find another service to do OCR, just as a fallback, so something is working if I don't figure out the filters fast enough.

I found an API, ocr.space, which allowed me to upload my picture and it would detect the text and return it, but it wasn't as reliable either. So, I decided to find and use a Gemini API, which worked every time. With this service, it was able to detect the text not solely based on the words, but also the context.
```bash
pip install google-genai
```

Soon, I just gave up on the filters and started working with the gemini api, focusing more on implementing more features.

At this point, I got a Google Books API and had code that took a picture from my Pi camera and uploaded it to Gemini, which extracted the text. Then, it sent the title and author to the Google Books API, which generated the top 5 results that matched the title entered. After that, I asked Gemini to generate 5 recommendations for the book.

For this code the users would have to use the terminal, which isn't very user friendly, so I was advised to make my own website.

In order to build my website, I used Streamlit—an open-source Python framework that allowed me to build the site using Python.
```bash
pip install streamlit
```

Once I had all the buttons set up to take picture and then get all the information you want about the book, I decided to create my own reccomendations algorithm.
I had looked at this github page for inspiration and guidance: [Book Recommendation System](https://github.com/vb8146649/Book-Recommend-System/tree/main?tab=readme-ov-file)

I began working in google colab notebook, where I was able to experiment freely and chunk up my code into different relevent sections. [My Notebook](https://colab.research.google.com/drive/1MSKweRVUYagDJmegP_dTD1MUTnL7cfKI?usp=sharing)

I originally wanted to use the author and the tags to recommend books, but I narrowed it down to just the tags. I used the Goodreads database: one file with tag IDs and tag names, one with tag IDs and book IDs, and the last one with book IDs, book names, authors, ratings, etc.

The basic idea of my algorithm is that it takes the title and author and tries to find the book in the database. Once it's found, it gets the book ID, then the tags associated with that book. It then compares those tags with the tags of every other book and gives each one a score based on how many tags they have in common. The top 5 highest-scoring books are returned as recommendations.

#### Outline
importing all the necessary packages
```python
import pandas as pd #allows me to create the set with the right columns and rows
```
loading all the datasets from GoodReads 10k project
```python
tags_url = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/tags.csv" #tag_id,tag_name
book_tags_url = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/book_tags.csv" #goodreads_book_id,tag_id,count
books_url = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/books.csv"#titles, authors, publication details, ratings, and cover image URLs
books = pd.read_csv(books_url)
tags = pd.read_csv(tags_url)
book_tags = pd.read_csv(book_tags_url)
```
Setting up the dataset with the right columns to help find similar books.
```python
book_tags_merged = pd.merge(book_tags, tags, on='tag_id') # Merge book_tags with tags
book_tags_set = book_tags_merged.groupby('goodreads_book_id')['tag_name'].agg(set).reset_index() #combines book id and the tags asscoiated with it

# Add book titles and authors by merging with books, so I can get author+title and get the id then get the tags
books_subset = books[['goodreads_book_id', 'title', 'authors']]
book_tags_set = pd.merge(book_tags_set, books_subset, on='goodreads_book_id')

# Rename and reorder columns
book_tags_set.rename(columns={'goodreads_book_id': 'book_id', 'tag_name': 'tags'}, inplace=True)
book_tags_set = book_tags_set[['book_id', 'title', 'authors', 'tags']]
```
recommend_books_by_title_author finds the book ID by title and author, then calls recommend_books to get recommendations based on tags.
```python
def recommend_books(book_id, book_tags_df, top_n=5):
    df = book_tags_df.copy()

    target_tags = df.loc[df['book_id'] == book_id, 'tags'].values
    if len(target_tags) == 0:
        print("Book ID not found.")
        return None

    target_tags = target_tags[0]

    def tag_overlap(row):
        return len(target_tags.intersection(row['tags']))

    df['overlap'] = df.apply(tag_overlap, axis=1)
    recommendations = df[df['book_id'] != book_id].sort_values(by='overlap', ascending=False).head(top_n)

    return recommendations[['book_id', 'title', 'tags', 'overlap']]

def recommend_books_by_title_author(title, author, book_tags_df, top_n=5):
    # Normalize title and author for matching
    title = title.strip().lower()
    author = author.strip().lower()

    # Find the matching book
    matched_book = book_tags_df[
        book_tags_set['title'].str.lower().str.contains(title) &
      book_tags_set['authors'].str.lower().str.contains(author)
    ]

    if matched_book.empty:
        print(f"No book found with title '{title}' and author '{author}'")
        return None

    book_id = matched_book.iloc[0]['book_id']
    target_title = matched_book.iloc[0]['title']
    print(f"\nFound Book: '{target_title}' (ID: {book_id}) — generating recommendations...\n")

    return recommend_books(book_id, book_tags_df, top_n=top_n)
```
Example run with <u>Harry Potter and the Prisoner of Azkaban</u>
```python
book_title = "Harry Potter and the Prisoner of Azkaban"
book_author = "J.K"

recommendations = recommend_books_by_title_author(book_title, book_author, book_tags_set, top_n=5)

if recommendations is not None:
    print(recommendations[['book_id', 'title', 'overlap']])
```


## Challenges

## Next Steps





# Second Milestone

<iframe width="560" height="315" src="https://www.youtube.com/embed/QT4CJaUCVO4?si=ClKCGyMrA-sgbFvg" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

<!--For your second milestone, explain what you've worked on since your previous milestone. You can highlight:
Technical details of what you've accomplished and how they contribute to the final goal
What has been surprising about the project so far
Previous challenges you faced that you overcame
What needs to be completed before your final milestone-->
## Summary
For my second milestone, I followed the steps from [Running TensorFlow Lite Object Recognition on the Raspberry Pi 4 or Pi 5](https://learn.adafruit.com/running-tensorflow-lite-on-the-raspberry-pi-4/tensorflow-2-setup) to detect objects using my Pi camera. After setting it up, I figured out that instead of detecting a bunch of objects unreliably, I want to detect text, specifically book titles and authors, to obtain data about that particular book. So, I downloaded OCR(optical character recognition) and spaCy's en_core_web_sm model. This setup allows me to extract text from images and then use NER(Named Entity Recognition) to identify and label titles and authors separately.


## Steps
#### Update the Raspberry Pi
```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y python3-pip
sudo apt install --upgrade -y python3-setuptools
```
Pip is a python package installer for python 3

#### Setup Virtual Environment
```bash
sudo apt install python3.11-venv
python -m venv env --system-site-packages
```
Ran commands that allow me to create a virtual environment in python - important because it allows me to run codes in an isolated environment
Creates env
Not understanding this caused a lot of problems due to global files interfering with files i was trying to install in virtual environments
```bash
source env/bin/activate
```
- activates the environment created


#### Upgrade Script
```bash
cd ~
sudo pip3 install --upgrade adafruit-python-shell
wget https://raw.githubusercontent.com/adafruit/Raspberry-Pi-Installer-Scripts/master/raspi-blinka.py
sudo python3 raspi-blinka.py
```
This caused a lot of errors about externally managed environments because I was an environment and i was using sudo which was trying to install it through the whole pi - so getting rid of sudo allowed it to run
Installs or upgrades the Adafruit Python Shell
Downloads the raspi-blinka.py
Necessary to control board
#### Tensor flow - Install requirements
```bash
sudo apt install -y python3-numpy python3-pillow python3-pygame
```
Downloaded 3 python packages that allow for fast computing(data processing), image processing, and making games with visuals and audio
```bash
sudo apt install -y festival
```
Speech package

#### Install rpi-vision
```bash
cd ~
source env/bin/activate
git clone --depth 1 https://github.com/adafruit/rpi-vision.git
cd rpi-vision
pip3 install -e .
```
Installing fork of adafruit program for detecting objects
#### Install TensorFlow 2.x
```bash
RELEASE=https://github.com/PINTO0309/Tensorflow-bin/releases/download/v2.15.0.post1/tensorflow-2.15.0.post1-cp311-none-linux_aarch64.whl
CPVER=$(python --version | grep -Eo '3\.[0-9]{1,2}' | tr -d '.')
pip install $(echo "$RELEASE" | sed -e "s/cp[0-9]\{3\}/CP$CPVER/g")
```
Installs tensorflow an open source library for machine learning - gives ability to run ai models to detect the objects 

#### Running the Graphic Labeling Demo
```bash
cd rpi-vision
python3 tests/pitft_labeled_output.py --tflite
```
Captures camera image and uses tflite(tensorflow lite) to do object detection - supposed to display to PiTFT screen but i use VNC to stream the video onto my computer - this makes my pi heat up to crazy temps so I added heat sinks
Also whatever is detected is outputted in audio form as well

After getting object detection set up - i realized that it is really really bad(probably due to camera quality or that the model has so many things to detect that it can’t do everything)
So I decided that I want to focus on detecting books(getting the title and author) then getting information about it and g=outputting in audio form to the user

I started by downloading OCR
I was only able to detect Hello from my iphone screen properly then i started using opencv to add filters - helping ocr to isolate words and detect them
Next i needed to separate author and book title so i used NER(named entity recognition which is a part of NLP natural language processing)
“Named Entity Recognition (NER) in NLP focuses on identifying and categorizing important information known as entities in text.”
Like people, places, dates, quantities

```bash
pip install spacy
pip install nltk
python -m spacy download en_core_web_sm
```

I had to install spacy which is an open source library for natural language processing, which allows computers to understand human languages. 
I also installed en_core_web_sm which is the english model for spacy

I also had a lot of problems in this part because i decided to use another environment to download all these packages because there was some problems with numPy and spacy so i created another environment and i forgot to switch interpreters so when running my code I got the same error of en_core_web_sm not being able to be used even though it was installed so I needed to switch interpreters in VScode and enter the right environment

At this point I just wanted to test if my setup worked without my pi cam so i took pictures from google of book covers and ran my code trying to detect the author with a person entity and the book title with heuristics(guessing/good enough)
I was using filters again but the problem this time is that some book covers aren’t clean enough and the words get blocked out - due to one section being darker and then when comparing the sections it blocks out the words instead of the background so i started just using the inside pages of books with the title and author on a blank page and it was able to separate the author and title for one picture perfectly so now my next steps are to clean up the detection for the author because the person detection isn’t always working and then adjusting filters until I’m able to use the book cover reliably

## Challenges
## Next Steps

# First Milestone

<iframe width="560" height="315" src="https://www.youtube.com/embed/jj_O0dZNq4Q?si=ML43E9GmnBaIR7hE" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

<!--For your first milestone, describe what your project is and how you plan to build it. You can include:
- An explanation about the different components of your project and how they will all integrate together
- Technical progress you've made so far
- Challenges you're facing and solving in your future milestones
- What your plan is to complete your project-->
## Summary
My first milestone was to set up my raspberry pi, connect a camera, and take a picture.
##### Figure 3: A picture taken from my pi camera
<img src="pi_cam_pic.png" style="width:30%; height:30%;">

## Steps
<!--The first thing I did was to flash the SD card. I replaced the preloaded 32 bit operating system with a 64 bit, allowing me to download all the libraries I would need for this project. After inserting the SD card into the Pi, I hooked my computer to my raspberry pi with a video capture card, allowing me to use my computer as a moniter, and enabled ssh(secure shell) in OBS. Secure shell allows the raspberry pi to be accessed remotely, while encrypting data through the network. The next step was to setup desktop setup by downloading tigervnc and vscode connecting it to my raspberry pi using ssh, so I can interface with my raspberry pi, run and edit code. -->

The first thing I did was flash the SD card. I replaced the preloaded 32-bit operating system with a 64-bit version, which allowed me to download all the libraries needed for this project. After inserting the SD card into the Raspberry Pi, I connected it to my computer using a video capture card, allowing me to use my computer as a monitor. I then enabled SSH (Secure Shell) in OBS. Secure Shell allows the Raspberry Pi to be accessed remotely while encrypting data transmitted over the network.

Next, I set up the desktop environment by installing TigerVNC and Visual Studio Code, connecting to the Raspberry Pi via SSH. This setup allows me to interface with the Pi and run and edit code directly from my computer. Now I'm able to use my raspberry pi without an external keyboard, mouse, and without the video capture.

Finally, I installed opencv:
```bash
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install libopencv-dev
sudo apt-get install python3-opencv
```
Making it easier to capture images and manipulate them.

This is the code I used to capture pictures.
``` python
from picamera2 import Picamera2, Preview
import time
import cv2
picam2 = Picamera2()
camera_config = picam2.create_still_configuration(main={"size": (1920, 1080)},
lores={"size": (640, 480)}, display="lores")
picam2.configure(camera_config)
#picam2.start_preview(Preview.QTGL) #Comment this out if not using desktop interface
picam2.start()
time.sleep(2)
im = picam2.capture_array()
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
cv2.imwrite('file.png', im)
```


## Challenges
Checked wifi connection with pinging the raspberry pi 
The host name wasn’t working so had to use the ip address
Basically just checks if the raspberry pi is there

ssh wasn't working due to network issues had to depend on Obs which 
So capture takes html video signals to usb signals so i can use my computer instead of a monitor then obs switches it back to hdmi signals and converts it to video we can see


## Next Steps

<!--# Schematics 
Here's where you'll put images of your schematics. [Tinkercad](https://www.tinkercad.com/blog/official-guide-to-tinkercad-circuits) and [Fritzing](https://fritzing.org/learning/) are both great resoruces to create professional schematic diagrams, though BSE recommends Tinkercad becuase it can be done easily and for free in the browser. -->

<!--# Code
Here's where you'll put your code. The syntax below places it into a block of code. Follow the guide [here]([url](https://www.markdownguide.org/extended-syntax/)) to learn how to customize it to your project needs. 

```bash
echo "Hello World"
```

```c++
void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  Serial.println("Hello World!");
}

void loop() {
  // put your main code here, to run repeatedly:

}
```-->

<!--# Bill of Materials
Here's where you'll list the parts in your project. To add more rows, just copy and paste the example rows below.
Don't forget to place the link of where to buy each component inside the quotation marks in the corresponding row after href =. Follow the guide [here]([url](https://www.markdownguide.org/extended-syntax/)) to learn how to customize this to your project needs. 

| **Part** | **Note** | **Price** | **Link** |
|:--:|:--:|:--:|:--:|
| Item Name | What the item is used for | $Price | <a href="https://www.amazon.com/Arduino-A000066-ARDUINO-UNO-R3/dp/B008GRTSV6/"> Link </a> |
| Item Name | What the item is used for | $Price | <a href="https://www.amazon.com/Arduino-A000066-ARDUINO-UNO-R3/dp/B008GRTSV6/"> Link </a> |
| Item Name | What the item is used for | $Price | <a href="https://www.amazon.com/Arduino-A000066-ARDUINO-UNO-R3/dp/B008GRTSV6/"> Link </a> |-->

# Starter Project: Jitterbug 


<iframe width="560" height="315" src="https://www.youtube.com/embed/-ZHt3RgyCjU?si=4k7KgGRrm4ysOyv0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

My starter project is a jitterbug. It consists of a vibration motor, 2 leds, battery, on and off switch, and 6 metal wires that serve as legs. When you switch the bug on, the motor will start shaking the legs, propelling the bug. 

##### Figure 2: The completed jitterbug
<img src="jitterbug.png" style="width:20%; height:20%;">

##### Figure 1: The circuit of the jitterbug
<img src="circuit.png" style="width:30%; height:30%;">



<!--# Other Resources/Examples
One of the best parts about Github is that you can view how other people set up their own work. Here are some past BSE portfolios that are awesome examples. You can view how they set up their portfolio, and you can view their index.md files to understand how they implemented different portfolio components.
- [Example 1](https://trashytuber.github.io/YimingJiaBlueStamp/)
- [Example 2](https://sviatil0.github.io/Sviatoslav_BSE/)
- [Example 3](https://arneshkumar.github.io/arneshbluestamp/)

To watch the BSE tutorial on how to create a portfolio, click here.-->
