# extract-unique-images-app
Python's streamlit-based web-app to store only unique images


Use the Python-Streamlit library to create a web-app that repeatedly triggers a camera (assume a request to a REST api), describes the image using a pre-trained neural network (use the small MobileNetV3 from torchvision) and compares it to stored images. 
When a button it pressed, create a new folder and save all distinct images in it (i.e. the MSE is below a configurable threshold).


## Structure

The repository is structured as follows:
``` 
BaslerCameraAdapter
+-- toydata  # contains pictures to demonstrate the app
+-- utils  # helper functions
|-- app.py  # streamlit app. This is the main file
|-- communication.py  # requests
|-- DataModels_BaslerCameraAdapter.py  # pydantic data models for requesting the camera adapter
|-- describe_images.py  # CNN (MobileNetV3) to extract features from an image
|-- ImageProcessingThread.py  # Threading class for parallelization
|-- LICENSE
|-- README.md
|-- requirements.txt  # pip requirements
|-- utils_image.py  # image manipulation functions
```

## Config
| environment variable              | type          | default                                    | description                                                                                                  |
|-----------------------------------|---------------|--------------------------------------------|--------------------------------------------------------------------------------------------------------------|
| LOGGING_LEVEL                     | int / str     | INFO                                       | logging level                                                                                                |
| TITLE                             | str / None    | "Extract unique images"                    | Text to display on top of the app                                                                            |
| INPUT_TEXT_TITLE                  | str / None    | "Order number"                             | Text to describe the test input                                                                              |
| INPUT_TEXT_PLACEHOLDER            | str / None    | "Please enter an order number here"        | Placeholder text of text input box                                                                           | 
| IMAGE_SAVE_QUALITY                | int           | 95                                         | JPEG quality in Percent                                                                                      |
| LOOP_UPDATE_TIME                  | float         | 0.1                                        | interval to update the app in seconds                                                                        |
| IMAGE_DISPLAY_WIDTH               | int           | 320                                        | Width of the image to display in pixels                                                                      |
| GALLERY_N_COLUMNS                 | int           | 6                                          | Number of columns that the gallery displays the stored images in                                             |
| GALLERY_OVERALL_HEIGHT            | None          |                                            | Limit the height of the gallery as a box in pixels                                                           |
| OPTIONS_IMAGE_RESOLUTION          | List[int]     | [240, 320, 480, 512, 640, 960, 1024, 1280] | List of options of the image resolution                                                                      |
| OPTIONS_IMAGE_RESOLUTION_INIT_IDX | int           | 2                                          | Initial entry                                                                                                |
| OPTIONS_SAVE_AS_CHOICE_INIT_IDX   | int           | 0                                          | Initial choice of the app mode                                                                               | 
| OPTIONS_TH_MIN                    | float         | 0.15                                       | Minimal threshold to select when identifying distinct images                                                 | 
| OPTIONS_TH_MAX                    | float         | 0.5                                        | Maximal threshold to select when identifying distinct images                                                 |
| OPTIONS_TH_INC                    | float         | 0.025                                      | Increment of the threshold to select when identifying distinct images                                        |
| OPTIONS_TH_INIT                   | float         | 0.325                                      | Initial value of threshold to select when identifying distinct images                                        |
| IMAGES_PER_SECOND                 | int           | 5                                          | Frequency of how images should be requested                                                                  |
| IMAGES_PER_SECOND_MIN             | int           | 1                                          | Minimal image request frequency                                                                              |
| IMAGES_PER_SECOND_MAX             | int           | 23                                         | Maximal image request frequency. It is advised to keep this well below 30                                    |
| VIDEO_CODEC                       | str / Literal | "h264"                                     | Codec that should be used to encode an video to. Options are: "mp4v", "h264", "X264", "avc1", "HEVC"]        |
| CAMERA_ADDRESS                    | str / None    | None                                       | URL to the REST api of a camera, e.g. https://github.com/max-scw/BaslerCameraAdapter                         |
| CAMERA_TOKEN                      | str / None    | None                                       | Access token if required                                                                                     |
| CAMERA_TIMEOUT                    | float         | 5                                          | Timeout in seconds for a camera request. Keep it above 1 second even if the frequency would not allow for it |
| TEST_DATA_FOLDER                  | str / None    | None                                       | Path to a folder with images to mimic camera requests. For debugging only                                    |