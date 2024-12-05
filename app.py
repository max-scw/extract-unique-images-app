import streamlit as st
from PIL import Image
from pathlib import Path
import logging
import numpy as np
from typing import List, Tuple
import cv2


from timeit import default_timer as timer
from time import sleep

from describe_images import describe_image, build_descriptor_model, calculate_feature_diffs

from typing import Literal


VideoCodec = Literal["mp4v", "h264", "X264", "avc1", "HEVC"]


KEY_OPTION_SAVE_DISTINCT_IMAGES = "distinct images"
KEY_OPTION_SAVE_VIDEO = "video"

# config  # TODO: make configurable
LOGGING_LEVEL = logging.DEBUG
GALLERY_N_COLUMNS: int = 6
GALLERY_IMAGE_WIDTH: int = 100

IMAGE_DISPLAY_WIDTH: int = 320
VIDEO_CODEC: VideoCodec = "h264"  # make sure that the openh264 codec is installed (version 1.8)

FRAMES_PER_SECOND: int = 10
TITLE: str = "Extract unique images"

IMAGE_RESOLUTION_OPTIONS = [240, 320, 480, 512, 640, 960, 1024, 1280]
IMAGE_RESOLUTION_IDX: int = 2

LAYOUT_LANDSCAPE: bool = False


logger = logging.getLogger(__name__)
logger.setLevel(LOGGING_LEVEL)


def save_as_video(files: List, output_video: Path):

    first_image_path = files[0]
    frame = cv2.imread(str(first_image_path))
    if frame is None:
        logger.warning(f"Unable to read image")
    height, width, _ = frame.shape
    frame_size = (width, height)
    fps = 3
    fourcc = cv2.VideoWriter_fourcc(*"WebM")
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, frame_size)

    for image_path in files:
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue

        video_writer.write(frame)

    video_writer.release()


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

def show_gallery():
    logger.debug(f"show_gallery(): writing {len(st.session_state.images_thumbnails)} images to {GALLERY_N_COLUMNS} columns")
    cols = st.columns(GALLERY_N_COLUMNS)
    for i, img in enumerate(st.session_state.images_thumbnails):
        cols[i % GALLERY_N_COLUMNS].image(img)
        # with cols[i % GALLERY_N_COLUMNS]:
        #     st.image(img, width=GALLERY_IMAGE_WIDTH)


def resize_image(img: Image.Image, max_width) -> Image.Image:
    """Resize an image to a maximum width while maintaining its aspect ratio."""
    # Calculate the new width and height while maintaining the aspect ratio
    width, height = img.size
    if width > max_width:
        new_width = max_width
        new_height = int(height * (max_width / width))
    else:
        new_width = width
        new_height = height

    # return the resized image
    return img.resize((new_width, new_height))


def change(mse_slider: float, image_files, img_feature_list):
    temp = []
    # Ensure we only iterate over indices that are valid for both lists
    for i in range(min(len(image_files), len(img_feature_list))):
        if img_feature_list[i] > mse_slider:
            temp.append(image_files[i])
    return temp


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

    if "image_resolution_idx" not in st.session_state:
        st.session_state.image_resolution_idx = IMAGE_RESOLUTION_IDX
    if "image_resolution" not in st.session_state:
        st.session_state.image_resolution = IMAGE_RESOLUTION_OPTIONS[IMAGE_RESOLUTION_IDX]

    if "video_writer" not in st.session_state:
        st.session_state.video_writer = None
    if "video_name" not in st.session_state:
        st.session_state.video_name = None


    # set flags
    keys_flag = [
        "running",
        "can_update",
        "button_stop",
        "button_show_video",
        "use_column_width"  # FIXME: what does this do?
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
                    options=(KEY_OPTION_SAVE_DISTINCT_IMAGES, KEY_OPTION_SAVE_VIDEO)
                )  # TODO: make default configurable

            with cols_options[1]:
                resolution = st.selectbox(
                    label="Image resolution",
                    options=IMAGE_RESOLUTION_OPTIONS,
                    index=st.session_state.image_resolution_idx,
                    help="Maximum length of the saves images or video frames in pixels."
                )
                logging.debug(f"Selected image resolution: {resolution}")
                # update session state
                st.session_state.image_resolution_idx = IMAGE_RESOLUTION_OPTIONS.index(resolution)
                st.session_state.image_resolution = resolution


            with cols_options[2]:
                th_mse = st.select_slider(
                    label="Threshold to determine distinct images",
                    options=[round(x, 3) for x in np.arange(0.2, 0.5, 0.025)],
                    value=st.session_state.threshold,  # default
                    key="slider_mse",
                    help="Threshold of the mean-squared-error (mse) between image features. "
                         "An image is saved if the minimal mse value to all stored images is larger than this threshold.",
                    disabled=choose != KEY_OPTION_SAVE_DISTINCT_IMAGES,
                )
                logging.debug(f"Selected threshold: {th_mse} (MSE)")
                # update session state
                st.session_state.threshold = th_mse

        # Layout: secondary controls
        with st.container():
            cols_control_secondary = st.columns([1, 1, 1, 1, 1], vertical_alignment="bottom", gap="small")
            with cols_control_secondary[4]:
                if choose == KEY_OPTION_SAVE_DISTINCT_IMAGES:
                    st.metric(
                        label="\# Images",
                        help="Unique images",
                        value=len(st.session_state.images_names)
                    )

            with cols_control_secondary[0]:
                if choose == KEY_OPTION_SAVE_DISTINCT_IMAGES:
                    button_save_image = st.button(
                        label="Save image",
                        key="button_save_image",
                        disabled=not st.session_state.running,
                        # use_container_width=True
                    )
                else:
                    button_save_image = False

    with space:
        # Layout: display messages
        message_container = st.container()
        # Layout: display
        display = st.container()

    if button_start:
        if not folder_name:
            message_container.warning(f"Please enter the folder name.")
        else:
            # reset message container
            message_container = st.empty()
            # reset display container
            # display = st.empty()
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
                diffs = calculate_feature_diffs(img_encoded, st.session_state.images_encoded)

                diff_min = min(diffs) if len(diffs) > 0 else 999

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
                        img.size
                    )
                # write frame to video
                st.session_state.video_writer.write(np.asarray(img))


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
            display.video(st.session_state.video_name.as_posix())

    if (not st.session_state.running) and (len(st.session_state.images_names) > 0):

        if choose == KEY_OPTION_SAVE_DISTINCT_IMAGES:
            with display:
                show_gallery()

    # if button_show:
    #     image_files = list(st.session_state.export_dir.glob("*.jpg"))
    #     message_container.success(f"Images have been in {st.session_state.folder_name} updated.")
    #     temp = change(th_mse, image_files, st.session_state.img_feature_list)
    #     for el in image_files:
    #         if el not in temp:
    #             Path(el).unlink()
    #     if len(temp) < 1:
    #         message_container.error("All images were deleted.")
    #     image_files = temp
    #     show_gallery()

    # if (not st.session_state.running) and (th_mse > st.session_state.threshold) and (not button_show):
    #
    #     image_files = list(st.session_state.export_dir.glob("*.jpg"))
    #     temp = change(th_mse, image_files, st.session_state.img_feature_list)
    #     if len(temp) > 1:
    #         show_gallery()
    #         message_container.success("slider changed")
    #         message_container.success(f"You get {len(temp)} images right now.")
    #     if len(temp) == 1:
    #         message_container.warning("Threshold set too high, only the first image remains.")
    #         show_gallery()
    # elif not st.session_state.running and th_mse < st.session_state.threshold:
    #     message_container.warning(f"No images were changed.")



if __name__ == "__main__":

    main()
