import streamlit as st
from pathlib import Path
import logging

import numpy as np

from timeit import default_timer as timer
from time import sleep


from DataModels_BaslerCameraAdapter import BaslerCameraSettings, ImageParams
from utils import get_env_variable, setup_logging
from ImageProcessingThread import ImageProcessingThread, get_min_difference

from typing import Literal, List, Tuple, Dict, Any


VideoCodec = Literal["mp4v", "h264", "X264", "avc1", "HEVC"]

KEY_OPTION_SAVE_DISTINCT_IMAGES = "distinct images"
KEY_OPTION_SAVE_VIDEO = "video"
SAVE_OPTIONS = (KEY_OPTION_SAVE_DISTINCT_IMAGES, KEY_OPTION_SAVE_VIDEO)


@st.cache_data
def get_config() -> Dict[str, Any]:
    # set up logging
    logger = setup_logging(__name__, logging.DEBUG)

    # set up config
    default_config = {
        "TITLE": "Extract unique images",
        "INPUT_TEXT_TITLE": "Order number",
        "INPUT_TEXT_PLACEHOLDER": "Please enter an order number here",
        "IMAGE_SAVE_QUALITY": 95,
        "LOOP_UPDATE_TIME": 0.1,
        # "LAYOUT_LANDSCAPE": False,
        "IMAGE_DISPLAY_WIDTH": 320,
        # gallery
        "GALLERY_N_COLUMNS": 6,
        "GALLERY_OVERALL_HEIGHT": None,
        # options
        "OPTIONS_IMAGE_RESOLUTION": [240, 320, 480, 512, 640, 960, 1024, 1280],
        "OPTIONS_IMAGE_RESOLUTION_INIT_IDX": 2,
        "OPTIONS_SAVE_AS_CHOICE_INIT_IDX": 0,
        "OPTIONS_TH_MIN": 0.15,
        "OPTIONS_TH_MAX": 0.5,
        "OPTIONS_TH_INC": 0.025,
        "OPTIONS_TH_INIT": 0.325,
        "IMAGES_PER_SECOND": 5,
        "IMAGES_PER_SECOND_MIN": 1,
        "IMAGES_PER_SECOND_MAX": 23,
        # video
        "VIDEO_CODEC": "h264",
        # camera
        "CAMERA_ADDRESS": None,
        "CAMERA_TOKEN": None,
        "CAMERA_TIMEOUT": 5,
        # DEBUGGING
        "TEST_DATA_FOLDER": None,
        # "TEST_DATA_FOLDER": Path(r"C:\Users\schwmax\OneDrive - Voith Group of Companies\SharedFiles\PackingDocumentation\Images") / "PXL_20241105_141202849 mit Schwenkarm.TS_30fps"
    }
    config = {ky: get_env_variable(ky, vl, check_for_prefix=True) for ky, vl in default_config.items()}

    config["CAMERA_SETTINGS"] = BaslerCameraSettings()
    config["CAMERA_IMAGE_PARAMETER"] = ImageParams()

    logger.debug(f"config: {config}")
    config["logger"] = logger
    return config


def set_running(value: bool):
    st.session_state.running = value


def get_images_exceeding_threshold() -> List[int]:
    list_stored_images = []
    for idx, img_enc in enumerate(st.session_state.images_encoded):
        diff_min = get_min_difference(img_enc, [st.session_state.images_encoded[el] for el in list_stored_images])
        if diff_min > st.session_state.threshold:
            list_stored_images.append(idx)
    return list_stored_images


def show_gallery(config: Dict[str, Any]) -> List[int]:
    logger = config["logger"]
    logger.debug(f"show_gallery(): len(st.session_state.images_thumbnails) {len(st.session_state.images_thumbnails)}")

    list_stored_images = get_images_exceeding_threshold()
    images = [st.session_state.images_thumbnails[el] for el in list_stored_images]

    logger.debug(f"show_gallery(): writing {len(images)} images to {config['GALLERY_N_COLUMNS']} columns")
    with st.container(height=config["GALLERY_OVERALL_HEIGHT"]):
        cols = st.columns(config["GALLERY_N_COLUMNS"])
        for i, img in enumerate(images):
            cols[i % config["GALLERY_N_COLUMNS"]].image(img)
    return list_stored_images


def initialize_session_state(config: Dict[str, Any]):
    
    if "threshold" not in st.session_state:
        st.session_state.threshold = config["OPTIONS_TH_INIT"]

    if "save_as" not in st.session_state:
        st.session_state.save_as = SAVE_OPTIONS[config["OPTIONS_SAVE_AS_CHOICE_INIT_IDX"]]

    # initialize lists
    key_list = ["images_thumbnails", "images_names", "images_encoded"]
    for ky in key_list:
        if ky not in st.session_state:
            setattr(st.session_state, ky, [])

    if "counter" not in st.session_state:
        st.session_state.counter = 0

    if "folder_head" not in st.session_state:
        st.session_state.folder_head = Path("./export")
        st.session_state.folder_head.mkdir(parents=True, exist_ok=True)

    if "folder_name" not in st.session_state:
        st.session_state.folder_name = ""

    if "image_resolution" not in st.session_state:
        st.session_state.image_resolution = config["OPTIONS_IMAGE_RESOLUTION"][config["OPTIONS_IMAGE_RESOLUTION_INIT_IDX"]]

    if "images_per_second" not in st.session_state:
        st.session_state.images_per_second = config["IMAGES_PER_SECOND"]

    if "thread" not in st.session_state:
        st.session_state.thread = None


    # set flags
    keys_flag = [
        "running",
        "button_stop",
    ]
    for ky in keys_flag:
        if ky not in st.session_state:
            setattr(st.session_state, ky, False)


def get_export_dir(name: str, folder: Path, make_dir: bool = True) -> Path:
    folder_ = Path(folder)

    i = 1
    name_ = name
    while (folder_ / name_).exists():
        name_ = f"{name}_{i}"
        i += 1

    export_dir = folder_ / name_

    if make_dir:
        export_dir.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger(__name__)
        logger.debug(f"New folder created: {export_dir}")
    return export_dir


def main():
    layout_landscape = get_env_variable("LAYOUT_LANDSCAPE", False)
    st.set_page_config(
        # page_title=get_env_variable("TITLE", "Extract unique images"),
        page_icon=":camera:",
        layout="wide" if layout_landscape else "centered"
    )  # must be called as the first Streamlit command in your script.

    config = get_config()
    logger = config["logger"]

    initialize_session_state(config)
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

    t0 = timer()

    st.title(config["TITLE"])

    # build layout: menu column layout
    menu, space = st.columns([1, 2]) if layout_landscape else st.container(), st.container()

    # put layout into effect
    with menu:
        # Layout: Controls
        with st.container():
            cols_control = st.columns([1, 3, 1], vertical_alignment="bottom", gap="small")
            with cols_control[0]:
                button_start = st.button(
                    label="Start",
                    disabled=st.session_state.running,
                    type="primary",
                )
            with cols_control[1]:
                folder_name = st.text_input(
                    label=config["INPUT_TEXT_TITLE"],
                    placeholder=config["INPUT_TEXT_PLACEHOLDER"],
                    disabled=st.session_state.running
                )
            with cols_control[2]:
                button_stop = st.button(
                    label="Close order",
                    key="stop_button",
                    disabled=not st.session_state.running,
                    on_click=lambda: set_running(False),
                    type="primary",
                    use_container_width=True
                )

        # Layout: Options
        with st.expander("Options"):
            cols_options = st.columns([1, 1, 1])

            with cols_options[0]:
                choose = st.radio(
                    label="Save as",
                    key="save_as",
                    options=SAVE_OPTIONS
                )

            with cols_options[1]:
                st.selectbox(
                    label="Image resolution",
                    options=config["OPTIONS_IMAGE_RESOLUTION"],
                    key="image_resolution",
                    help="Maximum length of the saves images or video frames in pixels.",
                    disabled=st.session_state.running
                )
                st.number_input(
                    label="Images per second",
                    min_value=config["IMAGES_PER_SECOND_MIN"],
                    max_value=config["IMAGES_PER_SECOND_MAX"],
                    step=1,
                    format="%d",
                    key="images_per_second",
                )

            with cols_options[2]:
                st.select_slider(
                    label="Threshold to determine distinct images",
                    options=np.arange(
                        config["OPTIONS_TH_MIN"],
                        config["OPTIONS_TH_MAX"] + 1e-10,
                        config["OPTIONS_TH_INC"]
                    ).round(3),
                    key="threshold",
                    help="Threshold of the mean-squared-error (mse) between image features. "
                         "An image is saved if the minimal mse value to all stored images is larger than this threshold.",
                    disabled=choose != KEY_OPTION_SAVE_DISTINCT_IMAGES,
                )

        # Layout: secondary controls
        with st.container():
            cols_control_secondary = st.columns([1, 1, 1, 1, 1], vertical_alignment="bottom", gap="small")
            with cols_control_secondary[0]:
                if choose == KEY_OPTION_SAVE_DISTINCT_IMAGES:
                    button_save_image = st.button(
                        label="Save image",
                        key="button_save_image",
                        disabled=not st.session_state.running,
                    )
                else:
                    button_save_image = False

            with cols_control_secondary[1]:
                if choose == KEY_OPTION_SAVE_DISTINCT_IMAGES:
                    button_update_images = st.button(
                        label="Update images",
                        key="button_update_images",
                        disabled=st.session_state.running or not st.session_state.images_names,
                    )
                else:
                    button_update_images = False

            with cols_control_secondary[4]:
                if choose == KEY_OPTION_SAVE_DISTINCT_IMAGES:
                    st.metric(
                        label="\# Images",
                        help="Unique images",
                        value=st.session_state.counter
                    )


    with space:
        # Layout: display messages
        message_container = st.container()
        # Layout: display
        _, display, _ = st.columns([0.1, 0.8, 0.1])
        # display = st.container()

    if button_start:
        if not folder_name:
            message_container.warning(f"Please enter the folder name.")
            st.session_state.running = False
        else:
            # reset message container
            message_container = st.empty()
            # create export directory
            st.session_state.export_dir = get_export_dir(name=folder_name, folder=st.session_state.folder_head)
            # reset loop variables
            st.session_state.running = True
            # reset image lists
            key_list = ["images_thumbnails", "images_names", "images_encoded", "image_date"]
            for ky in key_list:
                setattr(st.session_state, ky, [])
            # reset counter
            st.session_state.counter = 0
            # reset thread
            if st.session_state.thread is not None:
                st.session_state.thread.join()
                st.session_state.thread.stop()

            st.session_state.thread = ImageProcessingThread(
                address=config["CAMERA_ADDRESS"],
                camera_params=config["CAMERA_SETTINGS"],
                image_params=config["CAMERA_IMAGE_PARAMETER"],
                timeout=config["CAMERA_TIMEOUT"],
                token=config["CAMERA_TOKEN"],
                choose=choose,
                images_per_second=st.session_state.images_per_second,
                image_resolution=st.session_state.image_resolution,
                export_dir=st.session_state.export_dir,
                image_save_quality=config["IMAGE_SAVE_QUALITY"],
                video_codec=config["VIDEO_CODEC"],
                threshold=st.session_state.threshold,
                test_data_folder=config["TEST_DATA_FOLDER"],
            )
            st.session_state.thread.start()
            st.rerun()

    if st.session_state.thread is not None:
        st.session_state.thread.threshold = st.session_state.threshold
        st.session_state.thread.running = st.session_state.running
        st.session_state.thread.images_per_second = st.session_state.images_per_second

        if button_save_image:
            st.session_state.thread.save_image = True
        if (st.session_state.thread.image is not None) and st.session_state.running:
            display.image(st.session_state.thread.image)
        st.session_state.counter = len(st.session_state.thread.images_names)


    if button_stop:
        st.session_state.running = False

        thread = st.session_state.thread
        thread.running = False
        thread.join()
        thread.stop()

        st.session_state.video_name = thread.video_name
        st.session_state.images_names = thread.images_names
        st.session_state.images_encoded = thread.images_encoded
        st.session_state.images_thumbnails = thread.images_thumbnails

        if st.session_state.video_writer is not None:
            st.session_state.video_writer.release()
            logger.debug("Video writer released.")

        if (choose == KEY_OPTION_SAVE_DISTINCT_IMAGES) and (len(st.session_state.images_names) > 0):
            msg = f"{len(st.session_state.images_names)} image(s) saved."
            # message_container.success()
            st.toast(msg)
            display = st.empty()
        elif choose == KEY_OPTION_SAVE_VIDEO:
            with display:
                _, col, _ = st.columns([0.1, 0.8, 0.1])
                col.video(st.session_state.video_name.as_posix())

    elif st.session_state.running:
        dt = timer() - t0
        dt_wait = max([1 / st.session_state.images_per_second, config["LOOP_UPDATE_TIME"]]) - dt
        if dt_wait > 0:
            sleep(dt_wait)
        st.rerun()

    if (not st.session_state.running) and (len(st.session_state.images_names) > 0):
        if choose == KEY_OPTION_SAVE_DISTINCT_IMAGES:
            with display:
                idx_images = show_gallery(config)
                print(f"SHOW GALLERY {idx_images}")

    if button_update_images:
        image_names_old = st.session_state.images_names
        # update session_state
        key_list =["images_thumbnails", "images_names", "images_encoded"]
        for ky in key_list:
            value = getattr(st.session_state, ky)
            new_value = [value[el] for el in idx_images]
            setattr(st.session_state, ky, new_value)

        for nm in image_names_old:
            if nm not in st.session_state.images_names:
                fl = st.session_state.export_dir / nm
                logger.debug(f"Deleting image {fl.as_posix()}")
                fl.unlink()


if __name__ == "__main__":

    main()
