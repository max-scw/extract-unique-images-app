import streamlit as st
from PIL import Image
from pathlib import Path
from time import sleep
from timeit import default_timer
import logging
import numpy as np
from typing import List


from describe_images import describe_image, build_descriptor_model, calculate_feature_diffs

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def request_image(number: int) -> Image.Image | None:
    files = list(DATA_FOLDER.glob("*.jpg"))
    # load image from disk
    if number < len(files):
        return Image.open(files[number])
    else:
        st.success(f"All fotos were loaded. No more images left")
        return None


def set_running(value: bool):
    st.session_state.running = value
    if value is False:
        st.session_state.counter = 0
        st.session_state.image_counts = 0

def show_list_images(image_files: List):
    cols = st.columns(5)
    for i, image_path in enumerate(image_files):
        with cols[i % 5]:
            img = Image.open(image_path)
            st.image(img, caption=None, width=100)


def change(mse_slider: float, image_files, img_feature_list):
    temp = []
    for i, feature in enumerate(img_feature_list):
        if feature > mse_slider and i < len(img_feature_list):
            temp.append(image_files[i])
    return temp


def main():
    st.title("Extract distinct images")
    t0 = default_timer()

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

    if "img_feature_list" not in st.session_state:
        st.session_state.img_feature_list = []

    if "slider_default" not in st.session_state:
        st.session_state.slider_default = 0.325

    if "slider_tmp" not in st.session_state:
        st.session_state.slider_tmp = 0

    if "image_counts" not in st.session_state:
        st.session_state.image_counts = 0

    if "can_update" not in st.session_state:
        st.session_state.can_update = False

    # 0 means all the pictures
    save_folder = Path("data")

    message = st.container()
    cols = st.columns([1, 2, 1], vertical_alignment="bottom")
    with cols[0]:
        button_start = st.button("Start", icon="😃", disabled=st.session_state.running)
    with cols[1]:
        st.markdown(
            """
                    <style>
                    div[data-testid="stTextInput"] {
                        margin-left: -30px; 
                    }
                    </style>
                    """,
            unsafe_allow_html=True,
        )
        folder_name = st.text_input("Folder name")
    with cols[2]:
        button_save_image = st.button("Save this image", disabled=not st.session_state.running)

    cols2 = st.columns([1, 2, 1], vertical_alignment="bottom")
    button_stop = cols2[0].button(
        "Stop/Show", key="stop_button", disabled=not st.session_state.running, on_click=lambda: set_running(False)
    )

    export_dir = save_folder / folder_name
    image_files = list(list(export_dir.glob("*.png")) + list(export_dir.glob("*.jpg")))

    col5, col6 = st.columns([3, 1])

    with col5:
        mse_slider = st.select_slider(
            "Select a value of the MSE",
            options=[round(x, 3) for x in np.arange(0.2, 0.5, 0.025)],
            value=st.session_state.slider_default,  # default
            key="slider_mse",
        )

    with col6:
        st.metric(label="# Images", help="Unique images", value=st.session_state.image_counts)
    #    st.metric(value=st.session_state.counter)

    if button_start:
        if export_dir.exists():
            message.error(f"The folder '{folder_name}' already exists. Please choose a new name.")
            return
        st.session_state.running = True
        st.session_state.stored_images = []
        st.session_state.img_feature_list = []
        st.session_state.slider_move = 0

    if st.session_state.running:
        # create folder

        export_dir.mkdir(parents=True, exist_ok=True)

        container = st.container()

        logger.info(f"Starting loop. Saving images to {export_dir.as_posix()}")

        img = request_image(st.session_state.counter)

        if img is None:
            message.warning("No more images available, stopping.")
            st.session_state.running = False
            st.stop()
        else:

            # show image
            with container:
                st.write(img)

            # describe image
            img_encoded = describe_image(img, *st.session_state.model)

            diffs = calculate_feature_diffs(img_encoded, st.session_state.stored_images)

            diff_min = min(diffs) if len(diffs) > 0 else 999
            logger.debug(
                f"{st.session_state.counter}: minimal difference = {diff_min:.4g} to {len(st.session_state.stored_images)}"
            )
            if (diff_min > mse_slider) or button_save_image:
                # save image to folder
                # create file name

                filename = f"{st.session_state.image_counts}.jpg"
                img.save(export_dir / str(filename))
                logger.debug(f"Saved image to {export_dir / filename} (MSE: {diff_min:.4g} < {mse_slider:.4g})")
                st.session_state.img_feature_list.append(diff_min)
                # add to list of stored images (encoded)
                st.session_state.stored_images.append(img_encoded)
                st.session_state.image_counts += 1
                st.session_state.slider_tmp = mse_slider

            # increment counter
            st.session_state.counter += int(30 / FRAMES_PER_SECOND)
            dt = (1 / FRAMES_PER_SECOND) - (default_timer() - t0)
            if dt > 0:
                sleep(dt)
            # the length should check
            st.rerun()

    if folder_name == "":
        message.warning(f"Please enter the folder name.")

    if button_stop:
        st.session_state.can_update = True
        show_list_images(image_files)
        message.success(f"There were {len(image_files)} images saved.")

    button_update = st.button("Update", key="update_button", disabled=not st.session_state.can_update)

    if button_update:
        message.success("Images have been updated.")
        temp = change(mse_slider, image_files, st.session_state.img_feature_list)
        for el in image_files:
            if el not in temp:
                Path(el).unlink()
        if len(temp) < 1:
            message.error("All images were deleted.")

    if not st.session_state.running and mse_slider > st.session_state.slider_tmp:
        temp = change(mse_slider, image_files, st.session_state.img_feature_list)
        if len(temp) > 1:
            show_list_images(temp)
            message.success("slider changed")
            message.success(f"You get {len(temp)} images right now.")
        if len(temp) == 1:
            message.warning("Threshold set too high, only the first image remains.")
            show_list_images(temp)
    elif not st.session_state.running and mse_slider < st.session_state.slider_tmp:
        message.warning(f"This is the same images gallery.")
        temp = change(mse_slider, image_files, st.session_state.img_feature_list)
        show_list_images(temp)



if __name__ == "__main__":
    DATA_FOLDER = (
        Path(r"C:\Users\TianXue\OneDrive - Voith Group of Companies\PackingDocumentation\Images")
        / "PXL_20241105_141202849 mit Schwenkarm.TS_30fps"
    )
    TOTAL_COUNTS = len(list(DATA_FOLDER.glob("*.jpg")))
    FRAMES_PER_SECOND = 10
    main()
