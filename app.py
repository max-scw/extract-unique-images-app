import streamlit as st
from pathlib import Path
import logging

from PIL import Image
import cv2
import numpy as np

from timeit import default_timer as timer
from time import sleep

from describe_images import describe_image, build_descriptor_model, calculate_feature_diffs

from typing import Literal, List


VideoCodec = Literal["mp4v", "h264", "X264", "avc1", "HEVC"]


KEY_OPTION_SAVE_DISTINCT_IMAGES = "distinct images"
KEY_OPTION_SAVE_VIDEO = "video"

# config  # TODO: make configurable
LOGGING_LEVEL: int = logging.DEBUG
GALLERY_N_COLUMNS: int = 6
GALLERY_IMAGE_WIDTH: int = 100
GALLERY_OVERALL_HEIGHT: int | None = None

IMAGE_DISPLAY_WIDTH: int = 320
VIDEO_CODEC: VideoCodec = "h264"  # make sure that the openh264 codec is installed (version 1.8)

FRAMES_PER_SECOND: int = 10
TITLE: str = "Extract unique images"

IMAGE_RESOLUTION_OPTIONS: List[int] = [240, 320, 480, 512, 640, 960, 1024, 1280]
IMAGE_RESOLUTION_IDX: int = 2

LAYOUT_LANDSCAPE: bool = False


logger = logging.getLogger(__name__)
logger.setLevel(LOGGING_LEVEL)



def request_image() -> Image.Image | None:
    # FIXME: mockup
    data_folder = (
        Path(r"C:\Users\schwmax\OneDrive - Voith Group of Companies\SharedFiles\PackingDocumentation\Images")
        / "PXL_20241105_141202849 mit Schwenkarm.TS_30fps"
    )

    files = list(data_folder.glob("*.jpg"))
    # load image from disk
    if st.session_state.counter < len(files):
        img = Image.open(files[st.session_state.counter])  # FIXME: FRAMES_PER_SECOND
        st.session_state.counter += 1
        return img
    else:
        st.info(f"No more images left")
        return None


def set_running(value: bool):
    st.session_state.running = value
    if value is False:
        st.session_state.counter = 0


def get_min_difference(img_encoded, encoded_images: List[np.ndarray]) -> float:
    diffs = calculate_feature_diffs(img_encoded, encoded_images)

    return min(diffs) if len(diffs) > 0 else 999.


def get_images_exceeding_threshold() -> List[int]:
    list_stored_images = []
    for idx, img_enc in enumerate(st.session_state.images_encoded):
        diff_min = get_min_difference(img_enc, [st.session_state.images_encoded[el] for el in list_stored_images])
        if diff_min > st.session_state.threshold:
            list_stored_images.append(idx)
    return list_stored_images


def show_gallery() -> List[int]:
    logger.debug(f"show_gallery(): writing {len(st.session_state.images_thumbnails)} images to {GALLERY_N_COLUMNS} columns")

    list_stored_images = get_images_exceeding_threshold()
    images = [st.session_state.images_thumbnails[el] for el in list_stored_images]

    with st.container(height=GALLERY_OVERALL_HEIGHT):
        cols = st.columns(GALLERY_N_COLUMNS)
        for i, img in enumerate(images):
            cols[i % GALLERY_N_COLUMNS].image(img)
    return list_stored_images


def resize_image(img: Image.Image, max_length: int | float) -> Image.Image:
    """Resize an image to a maximum width while maintaining its aspect ratio."""
    # Calculate the new width and height while maintaining the aspect ratio
    width, height = img.size  # PIL: (width, height)

    len_max = max(img.size)
    len_min = min(img.size)

    if len_max > max_length:
        len_max_new = max_length
        len_min_new = int(len_min * (max_length / len_max))
    else:
        len_max_new = len_max
        len_min_new = len_min

    size_new = (len_max_new, len_min_new) if width > height else (len_min_new, len_max_new)

    # return the resized image
    return img.resize(size_new)


def initialize_session_state():

    if "model" not in st.session_state:
        model, preprocess = build_descriptor_model()
        st.session_state.model = (model, preprocess)
    
    if "threshold" not in st.session_state:
        st.session_state.threshold = 0.325


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

    # if "image_resolution_idx" not in st.session_state:
    #     st.session_state.image_resolution_idx = IMAGE_RESOLUTION_IDX
    if "image_resolution" not in st.session_state:
        st.session_state.image_resolution = IMAGE_RESOLUTION_OPTIONS[IMAGE_RESOLUTION_IDX]

    if "video_writer" not in st.session_state:
        st.session_state.video_writer = None
    if "video_name" not in st.session_state:
        st.session_state.video_name = None


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
        logger.debug(f"New folder created: {export_dir}")
    return export_dir


def main():
    st.set_page_config(
        page_title=TITLE,
        page_icon=":camera:",
        layout="wide" if LAYOUT_LANDSCAPE else "centered"
    )  # must be called as the first Streamlit command in your script.

    initialize_session_state()
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

    st.title(TITLE)

    # build layout: menu column layout
    menu, space = st.columns([1, 2]) if LAYOUT_LANDSCAPE else st.container(), st.container()

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
                    label="Order number",  # TODO: make configurable
                    placeholder="Please enter an order number here",
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
                    options=(KEY_OPTION_SAVE_DISTINCT_IMAGES, KEY_OPTION_SAVE_VIDEO)
                )  # TODO: make default configurable

            with cols_options[1]:
                st.selectbox(
                    label="Image resolution",
                    options=IMAGE_RESOLUTION_OPTIONS,
                    key="image_resolution",
                    help="Maximum length of the saves images or video frames in pixels.",
                    disabled=st.session_state.running
                )
                logging.debug(f"Selected image resolution: {st.session_state.image_resolution}")



            with cols_options[2]:
                st.select_slider(
                    label="Threshold to determine distinct images",
                    options=[round(x, 3) for x in np.arange(0.15, 0.5001, 0.025)],
                    key="threshold",
                    help="Threshold of the mean-squared-error (mse) between image features. "
                         "An image is saved if the minimal mse value to all stored images is larger than this threshold.",
                    disabled=choose != KEY_OPTION_SAVE_DISTINCT_IMAGES,
                )
                logging.debug(f"Selected threshold: {st.session_state.threshold} (MSE)")

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
                        value=len(st.session_state.images_names)
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
            # reset video writer
            st.session_state.video_writer = None

    if st.session_state.running:

        img = request_image()
        img = resize_image(img, st.session_state.image_resolution)
        if img is None:
            message_container.warning("No image retrieved. You may want to stop the mode.")
        else:
            # display image
            display.image(img, width=IMAGE_DISPLAY_WIDTH)

            if choose == KEY_OPTION_SAVE_DISTINCT_IMAGES:
                # encode image
                img_encoded = describe_image(img, *st.session_state.model)
                diff_min = get_min_difference(img_encoded, st.session_state.images_encoded)

                if (diff_min > st.session_state.threshold) or button_save_image:
                    logger.debug(f"{st.session_state.counter}: minimal difference = {diff_min:.4g} > {st.session_state.threshold}")
                    # save image to folder
                    filename = f"{len(st.session_state.images_names)}.jpg"
                    filepath = st.session_state.export_dir / filename
                    img.save(filepath)

                    # store image
                    st.session_state.images_encoded.append(img_encoded)
                    st.session_state.images_names.append(filename)
                    st.session_state.images_thumbnails.append(img)

                    logger.debug(f"Image saved to {filepath} (minimal MSE: {diff_min:.4g})")
            elif choose == KEY_OPTION_SAVE_VIDEO:
                if st.session_state.video_writer is None:
                    # initialize video writer
                    fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
                    st.session_state.video_name = st.session_state.export_dir / f"{st.session_state.export_dir.stem}.mp4"
                    st.session_state.video_writer = cv2.VideoWriter(
                        st.session_state.video_name.as_posix(),
                        fourcc,
                        FRAMES_PER_SECOND,
                        img.size  # PIL: (width, height)
                    )
                    logger.debug(
                        f"Set up a VideoWriter for {st.session_state.video_name.as_posix()} "
                        f"(codec={VIDEO_CODEC}, fps={FRAMES_PER_SECOND}, size={img.size})"
                    )
                # write frame to video
                st.session_state.video_writer.write(np.asarray(img, dtype=np.uint8))

            else:
                raise Exception(f"Unexpected option to save images: {choose}")

            # wait
            dt = timer() - t0
            dt_wait = (1 / FRAMES_PER_SECOND) - dt
            logger.debug(f"Sleep {dt:.4} seconds")
            if dt_wait > 0:
                sleep(dt_wait)
            st.rerun()

    if button_stop:
        st.session_state.button_stop = True
        st.session_state.running = False

        if st.session_state.video_writer is not None:
            st.session_state.video_writer.release()
            logger.debug("Video writer released.")

        if (choose == KEY_OPTION_SAVE_DISTINCT_IMAGES) and (len(st.session_state.images_names) > 0):
            msg = f"{len(st.session_state.images_names)} image(s) saved."
            # message_container.success()
            st.toast(msg)
        elif choose == KEY_OPTION_SAVE_VIDEO:
            with display:
                _, col, _ = st.columns([0.1, 0.8, 0.1])
                col.video(st.session_state.video_name.as_posix())

    if (not st.session_state.running) and (len(st.session_state.images_names) > 0):

        if choose == KEY_OPTION_SAVE_DISTINCT_IMAGES:
            with display:
                idx_images = show_gallery()

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
