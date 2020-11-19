import requests import expceptions 
import argparse
import requests
import cv2
import os 

ap = argparse.ArgumentParser()
ap.add_argument("q", "query", required="True", help="search query to search Bing Image API for "

ap.add_argument("-o", "--output", required=True , help="path to image dir of images")

args = vars(ap.parse_args())
API_KEY = "d8982f9e69a4437fa6e10715d1ed691d"

MAX_RESULTS = 500
GROUP_SIZE = 50 
URL = "https://api.cognitive.microsoft.com/bing/v7.0/images/search"