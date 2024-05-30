from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry
from msrest.authentication import ApiKeyCredentials
import cv2
import os
import time
import uuid

# Retrieve environment variables
ENDPOINT = ""
PREDICTION_ENDPOINT = ""
training_key = ""
prediction_key = ""
prediction_resource_id = ""

# Set up the training credentials
credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
trainer = CustomVisionTrainingClient(ENDPOINT, credentials)

# Set up the prediction credentials
prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(PREDICTION_ENDPOINT, prediction_credentials)

publish_iteration_name = "classifyModel"

# Project name
project_name = "face-recognition-dev2"

# Check if the project exists
existing_projects = trainer.get_projects()
project = None

for proj in existing_projects:
    if proj.name == project_name:
        project = proj
        print(f"Project '{project_name}' already exists.")
        break

if not project:
    # Create a new project if it doesn't exist
    print("Creating project...")
    project = trainer.create_project(project_name)
    print(f"Project '{project_name}' created.")

# Function to create a tag if it does not exist
def get_or_create_tag(project_id, tag_name):
    tags = trainer.get_tags(project_id)
    for tag in tags:
        if tag.name == tag_name:
            print(f"Tag '{tag_name}' already exists.")
            return tag
    print(f"Creating tag '{tag_name}'...")
    return trainer.create_tag(project_id, tag_name)

# Make tags in the project or use existing ones
hemlock_tag = get_or_create_tag(project.id, "Hemlock")
cherry_tag = get_or_create_tag(project.id, "Japanese Cherry")

base_image_location = os.path.join(os.path.dirname(__file__), "Images")

print("Adding images...")

image_list = []

for image_num in range(1, 11):
    file_name = "hemlock_{}.jpg".format(image_num)
    with open(os.path.join(base_image_location, "Hemlock", file_name), "rb") as image_contents:
        image_list.append(ImageFileCreateEntry(name=file_name, contents=image_contents.read(), tag_ids=[hemlock_tag.id]))

for image_num in range(1, 11):
    file_name = "japanese_cherry_{}.jpg".format(image_num)
    with open(os.path.join(base_image_location, "Japanese_Cherry", file_name), "rb") as image_contents:
        image_list.append(ImageFileCreateEntry(name=file_name, contents=image_contents.read(), tag_ids=[cherry_tag.id]))


print("Training...")
try:
    iteration = trainer.train_project(project.id)
    while iteration.status != "Completed":
        iteration = trainer.get_iteration(project.id, iteration.id)
        print("Training status: " + iteration.status)
        print("Waiting 10 seconds...")
        time.sleep(10)
    # The iteration is now trained or reused. Publish it to the project endpoint
    trainer.publish_iteration(project.id, iteration.id, publish_iteration_name, prediction_resource_id)
    iteration = trainer.get_iterations(project.id)[-1]
    print("Done!")
except Exception as e:
    if "Nothing changed since last training" in str(e):
        print("No new data to train on. Using the existing iteration.")
        iteration = trainer.get_iterations(project.id)[-1]
    else:
        raise



# Now there is a trained endpoint that can be used to make a prediction
prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(PREDICTION_ENDPOINT, prediction_credentials)

with open(os.path.join (base_image_location, "Test/test_image.jpg"), "rb") as image_contents:
    results = predictor.classify_image(
       project.id, iteration.publish_name, image_contents.read())

    # Display the results.
    for prediction in results.predictions:
        print("\t" + prediction.tag_name +
              ": {0:.2f}%".format(prediction.probability * 100))


