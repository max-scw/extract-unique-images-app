import streamlit as st

from PIL import Image

from pathlib import Path
from time import sleep
from datetime import datetime
import logging

from describe_images import describe_image, build_descriptor_model, calculate_feature_diffs

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def request_image(number: int) -> Image:
    files = list(Path("toydata").glob("*.jpg"))
    # load image from disk
    return Image.open(files[number])


def set_running(value: bool):
    st.session_state.running = value
    if value is False:
        st.session_state.counter = 0


def main():
    st.title("Extract distinct images")

    if "running" not in st.session_state:
        st.session_state.running = False

    if "model" not in st.session_state:
        model, preprocess = build_descriptor_model()
        st.session_state.model = (model, preprocess)

    if "stored_images" not in st.session_state:
        st.session_state.stored_images = []

    if "counter" not in st.session_state:
        st.session_state.counter = 0
    logger.debug("Session state variables initialized.")

    th_mse = 0.1  # FIXME: make optional user input
    #0 means all of the pictures
    save_folder = Path("data")

    cols = st.columns([1, 2, 1])
    with cols[0]:
        button_start = st.button("Start", disabled=st.session_state.running)
    with cols[1]:
        folder_name = st.text_input("Folder name")
    with cols[2]:
        button_stop = st.button("Stop", disabled=not st.session_state.running, on_click=lambda: set_running(False))

    if button_start:
        st.session_state.running = True
    if st.session_state.running:
        # create folder
        export_dir = save_folder / folder_name
        export_dir.mkdir(parents=True, exist_ok=True)

        container = st.container()

        st.session_state.stored_images = []
        logger.info(f"Starting loop. Saving images to {export_dir.as_posix()}")

        img = request_image(st.session_state.counter)
        # show image
        with container:
            st.write(img)
        # describe image
        img_encoded = describe_image(img, *st.session_state.model)

        diffs = calculate_feature_diffs(img_encoded, st.session_state.stored_images)
        diff_min = min(diffs) if len(diffs) > 0 else 999
        logger.debug(f"{st.session_state.counter}: minimal difference = {diff_min:.4g} to {len(st.session_state.stored_images)}")
        if diff_min > th_mse:
            # save image to folder
            # create file name
            time_string = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{time_string}.jpg"

            img.save(export_dir / filename)
            logger.debug(f"Saved image to {export_dir / filename} (MSE: {diff_min:.4g} < {th_mse:.4g})")

            # add to list of stored images (encoded)
            st.session_state.stored_images.append(img_encoded)

            if button_stop:
                st.session_state.running = False

        # increment counter
        st.session_state.counter += 1
        #
        sleep(1)
        st.rerun()


if __name__ == "__main__":
    main()
