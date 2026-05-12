# cs682-final-project
Final project for neural networks and computer vision. In this project we explore the use of computer vision models for predicting localizd poverty rates in US cities.

## Directory Details

This data contains code both from our experiments, as well as cloned code  from https://github.com/axin1301/Satellite-imagery-dataset/tree/main in order to download the images and poverty data from the dataset

- Data Collection
    - filtered_cities.csv: maps CBGs to specific cities and poverty data
    - data_processing.py: filtering operations
    - image_to_cbg: maps image filenames to specific CBGs
    - satellite_imagery_collection/
        - data_dir: Where downloaded images are placed
        - tilefile_scd: data for downloading specific images by spatial location
        - download_image.py: run to actually download images
    - labels: Contains image filename, CBG, and poverty rate for each city

- Training and evaluation 
    - experiments/: code for running training and evaluation
    - logs/: contains log files of experimental runs on Unity platform
    - qualitative: contains images from qualitative saliency map and attention tiles

## Other notes
- We intented to also include data from Dallas and Jacksonville in our model training, but due to a technical mistake and time constraints did not add those cities to our training. This repository has references to these two cities, but they were not reflected in the actual training/evaluation of our models/



