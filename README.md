# extract-unique-images-app
Python's streamlit-based web-app to store only unique images


Use the Python-Streamlit library to create a web-app that repeatedly triggers a camera (assume a request to a REST api), describes the image using a pre-trained neural network (use the small MobileNetV3 from torchvision) and compares it to stored images. 
When a button it pressed, create a new folder and save all distinct images in it (i.e. the MSE is below a configurable threshold).


## Structure

The repository is structured as follows:
``` 
BaslerCameraAdapter
+-- toydata  <- contains pictures to demonstrate the app
|-- LICENSE
|-- README.md
|-- requirements.txt  <- pip requirements
```