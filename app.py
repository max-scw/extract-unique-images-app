import streamlit as st
from PIL import Image
from pathlib import Path
from time import sleep
from timeit import default_timer
import logging
import numpy as np
from typing import List, Tuple


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


def show_list_images(image_files: List, resolution_slider: Tuple):
    cols = st.columns(3)
    for i, image_path in enumerate(image_files):
        with cols[i % 3]:
            img = Image.open(image_path)
            img_resized = img.resize(resolution_slider)
            st.image(img_resized, caption=None, width=200)


def change(mse_slider: float, image_files, img_feature_list):
    temp = []
    # Ensure we only iterate over indices that are valid for both lists
    for i in range(min(len(image_files), len(img_feature_list))):
        if img_feature_list[i] > mse_slider:
            temp.append(image_files[i])
    return temp


def main():
    st.title("Packing Documentation")
    t0 = default_timer()

    if "running" not in st.session_state:
        st.session_state.running = False

    if "model" not in st.session_state:
        model, preprocess = build_descriptor_model()
        st.session_state.model = (model, preprocess)

    # Feature values of all opened images
    if "stored_images" not in st.session_state:
        st.session_state.stored_images = []

    if "counter" not in st.session_state:
        st.session_state.counter = 0
    logger.debug("Session state variables initialized.")

    # Feature values of the images stored in the target folder
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

    if "folder_name" not in st.session_state:
        st.session_state.folder_name = ""

    # 0 means all the pictures
    save_folder = Path("data")

    message = st.container()
    cols = st.columns([1, 2, 1, 1], vertical_alignment="bottom")
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
        folder_name = st.text_input("Enter order number")
    with cols[2]:
        button_stop = st.button(
            "Close order", key="stop_button", disabled=not st.session_state.running, on_click=lambda: set_running(False)
        )

    export_dir = save_folder / folder_name
    image_files = list(list(export_dir.glob("*.png")) + list(export_dir.glob("*.jpg")))
    with st.expander("Options"):
        col3, col4 = st.columns([2, 1])
        with col3:
            choose = st.radio("Save as", ("as distinct images", "as video"))
        with col4:
            if choose == "as distinct images":
                button_save_image = st.button("Save image", disabled=not st.session_state.running)

        col5, col6 = st.columns([3, 1])

        with col5:
            mse_slider = st.select_slider(
                "Threshold to determine distinct images",
                options=[round(x, 3) for x in np.arange(0.2, 0.5, 0.025)],
                value=st.session_state.slider_default,  # default
                key="slider_mse",
            )

        with col6:
            if choose == "as distinct images":
                st.metric(label="# Images", help="Unique images", value=st.session_state.image_counts)

        selected_resolution_str = st.select_slider(
            "Image resolution (to display in pixels of maximum side)",
            options=["640x480", "1280x720", "1920x1080"],
            value="1280x720",
        )
    resolution_slider = tuple(map(int, selected_resolution_str.split("x")))

    if button_start:
        st.session_state.running = True
        st.session_state.stored_images = []
        st.session_state.img_feature_list = []
        st.session_state.image_counts = 0
        st.session_state.folder_name = folder_name
        export_dir = save_folder / folder_name

        if folder_name == "":
            message.warning(f"Please enter the folder name.")
            st.stop()
        else:
            while export_dir.exists():
                if "_" in folder_name and folder_name.split("_")[-1].isdigit():
                    base_name = "_".join(folder_name.split("_")[:-1])
                    count = int(folder_name.split("_")[-1]) + 1
                    folder_name = f"{base_name}_{count}"
                else:
                    folder_name = f"{folder_name}_1"
                export_dir = save_folder / folder_name

            export_dir.mkdir(parents=True, exist_ok=True)
            st.session_state.folder_name = folder_name
            logger.info(f"New folder created: {export_dir}")
            message.success(f"Folder '{folder_name}' created successfully.")

    if st.session_state.running:
        # to ensure that the export_dir has changed
        export_dir = save_folder / st.session_state.folder_name
        export_dir.mkdir(parents=True, exist_ok=True)

        container = st.container()
        logger.info(f"Starting loop. Saving images to {export_dir.as_posix()}")

        img = request_image(st.session_state.counter)
        if img is None:
            message.warning("No more images available, stopping.")
            st.session_state.running = False
            st.stop()
        else:
            with container:
                st.image(img, caption="Current Image", use_column_width=True)

            img_encoded = describe_image(img, *st.session_state.model)
            diffs = calculate_feature_diffs(img_encoded, st.session_state.stored_images)

            diff_min = min(diffs) if len(diffs) > 0 else 999
            logger.debug(f"{st.session_state.counter}: minimal difference = {diff_min:.4g}")

            if ((diff_min > mse_slider) or button_save_image) and choose == "as distinct images":
                filename = f"{st.session_state.image_counts}.jpg"
                img.save(export_dir / filename)
                logger.debug(f"Image saved to {export_dir / filename} (MSE: {diff_min:.4g})")

                st.session_state.stored_images.append(img_encoded)
                st.session_state.img_feature_list.append(diff_min)
                st.session_state.image_counts += 1
                st.session_state.slider_tmp = mse_slider

            st.session_state.counter += int(30 / FRAMES_PER_SECOND)
            st.rerun()

    if button_stop:
        export_dir = save_folder / st.session_state.folder_name
        image_files = list(export_dir.glob("*.jpg"))
        st.session_state.can_update = True
        show_list_images(image_files, resolution_slider)
        message.success(f"There were {len(image_files)} images saved in folder '{st.session_state.folder_name}'.")

    button_update = st.button("Update", key="update_button", disabled=not st.session_state.can_update)

    if button_update:
        export_dir = save_folder / st.session_state.folder_name
        image_files = list(export_dir.glob("*.jpg"))
        message.success(f"Images have been in {st.session_state.folder_name} updated.")
        temp = change(mse_slider, image_files, st.session_state.img_feature_list)
        for el in image_files:
            if el not in temp:
                Path(el).unlink()
        if len(temp) < 1:
            message.error("All images were deleted.")
        image_files = temp
        show_list_images(image_files, resolution_slider)

    if not st.session_state.running and mse_slider > st.session_state.slider_tmp and not button_update:
        export_dir = save_folder / st.session_state.folder_name
        image_files = list(export_dir.glob("*.jpg"))
        temp = change(mse_slider, image_files, st.session_state.img_feature_list)
        if len(temp) > 1:
            show_list_images(temp, resolution_slider)
            message.success("slider changed")
            message.success(f"You get {len(temp)} images right now.")
        if len(temp) == 1:
            message.warning("Threshold set too high, only the first image remains.")
            show_list_images(temp, resolution_slider)
    elif not st.session_state.running and mse_slider < st.session_state.slider_tmp:
        message.warning(f"This is the same images gallery.")
        temp = change(mse_slider, image_files, st.session_state.img_feature_list)
        show_list_images(temp, resolution_slider)


if __name__ == "__main__":
    DATA_FOLDER = (
        Path(r"C:\Users\TianXue\OneDrive - Voith Group of Companies\PackingDocumentation\Images")
        / "PXL_20241105_141202849 mit Schwenkarm.TS_30fps"
    )
    TOTAL_COUNTS = len(list(DATA_FOLDER.glob("*.jpg")))
    FRAMES_PER_SECOND = 10
    main()
