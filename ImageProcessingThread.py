import threading

import cv2
import numpy as np
from PIL import Image

from pathlib import Path
from timeit import default_timer as timer
from time import sleep

from typing import List

from utils import setup_logging
# from app import KEY_OPTION_SAVE_VIDEO, KEY_OPTION_SAVE_DISTINCT_IMAGES

from communication import build_url, request_camera
from utils_image import resize_image, bytes_to_image_pil
from describe_images import describe_image, build_descriptor_model, calculate_feature_diffs
from DataModels_BaslerCameraAdapter import BaslerCameraSettings, ImageParams

KEY_OPTION_SAVE_DISTINCT_IMAGES = "distinct images"  # FIXME: import options from somewhere
KEY_OPTION_SAVE_VIDEO = "video"
SAVE_OPTIONS = (KEY_OPTION_SAVE_DISTINCT_IMAGES, KEY_OPTION_SAVE_VIDEO)


logger = setup_logging(__name__)


def get_min_difference(img_encoded, encoded_images: List[np.ndarray]) -> float:
    diffs = calculate_feature_diffs(img_encoded, encoded_images)

    return min(diffs) if len(diffs) > 0 else 999.


class ImageProcessingThread(threading.Thread):
    def __init__(
            self,
            address: str,
            camera_params: BaslerCameraSettings,
            image_params: ImageParams,
            timeout: int,
            images_per_second: int,
            image_resolution,
            choose: str,
            export_dir: Path = None,
            threshold: float = 0.1,
            video_codec: str | None = None,
            image_save_quality: int | None = 95,
            token: str | None = None,
            test_data_folder: Path | str | None = None,
    ):
        super().__init__()
        # store local / private variables
        self.image_resolution = image_resolution
        self.threshold: float = threshold
        self.choose: str = choose
        self.export_dir: Path = Path(export_dir) if isinstance(export_dir, (str, Path)) else None
        self.images_per_second: int = images_per_second
        self.video_codec: str | None = video_codec
        self.image_save_quality: int | None = image_save_quality
        self.timeout: int = timeout

        # camera
        self.url: str = build_url(address, camera_params, image_params) if address else None
        self.__token: str | None = token
        self.test_files: List[Path] | None = list(Path(test_data_folder).glob("*.jpg")) if test_data_folder else None

        # set up model to identify distinct images
        # if self.choose == KEY_OPTION_SAVE_DISTINCT_IMAGES
        self._model, self._preprocess = build_descriptor_model()
        self.images_encoded: List[np.ndarray] = []
        self.images_names: List[str] = []
        self.images_thumbnails: List[Image.Image] = []
        # set up video writer
        self.video_writer = None
        self.video_name: Path = self.export_dir / f"{self.export_dir.stem}.mp4"
        # internal flags
        self._counter: int = 0
        self.image: Image.Image | None = None
        self.running: bool = True
        self.save_image: bool = True

    def request_image(self) -> Image.Image | None:
        t0 = timer()
        img = None
        if self.test_files is None:
            content = request_camera(self.url, self.timeout, token=self.__token)
            if content is not None:
                img = bytes_to_image_pil(content)
        else:
            if self._counter < len(self.test_files):
                img = Image.open(self.test_files[self._counter])
                self._counter += 1

        logger.debug(f"trigger_camera(): request_camera(url={self.url}, timeout={self.timeout}) (took {(timer() - t0) * 1000:.2g} ms)")
        return img

    def run(self):
        while self.running:
            t0 = timer()
            img = self.request_image()

            if img is not None:
                img = resize_image(img, self.image_resolution)
                self.image = img
            #     message_container.warning("No image retrieved. You may want to stop the mode.")
            # else:
                if self.choose == KEY_OPTION_SAVE_DISTINCT_IMAGES:
                    # encode image
                    img_encoded = describe_image(img, self._model, self._preprocess)
                    diff_min = get_min_difference(img_encoded, self.images_encoded)

                    if (diff_min > self.threshold) or self.save_image:
                        logger.debug(f"{self._counter}: minimal difference = {diff_min:.4g} > {self.threshold}")
                        # save image to folder
                        filename = f"{len(self.images_names)}.jpg"
                        filepath = self.export_dir / filename
                        threading.Thread(
                            target=img.save,
                            args=(filepath, ),
                            kwargs={"quality": self.image_save_quality}
                        ).start()
                        # img.save(filepath, quality=self.image_save_quality)

                        # store image
                        self.images_encoded.append(img_encoded)
                        self.images_names.append(filename)
                        self.images_thumbnails.append(img)

                        self.save_image = False

                        logger.debug(f"Image saved to {filepath} (minimal MSE: {diff_min:.4g})")
                elif self.choose == KEY_OPTION_SAVE_VIDEO:
                    if self.video_writer is None:
                        # initialize video writer
                        fourcc = cv2.VideoWriter_fourcc(*self.video_codec)
                        self.video_writer = cv2.VideoWriter(
                            self.video_name.as_posix(),
                            fourcc,
                            self.images_per_second,
                            img.size  # PIL: (width, height)
                        )
                        logger.debug(
                            f"Set up a VideoWriter for {self.video_name.as_posix()} "
                            f"(codec={self.video_codec}, fps={self.images_per_second}, size={img.size})"
                        )
                    # write frame to video
                    self.video_writer.write(np.asarray(img, dtype=np.uint8))

            dt = timer() - t0
            dt_des = 1 / self.images_per_second
            dt_wait = dt_des - dt
            if dt_wait > 0:
                sleep(dt_wait)
            else:
                logger.warning(f"{dt:.4} seconds too slow! Execution took {dt:.4g} seconds (should {dt_des:.4g}s).")

    def stop(self):
        """Stops the thread."""
        self.run = False
        logger.info("Stopping camera thread.")
        if self.video_writer is not None:
            self.video_writer.release()
            logger.debug("Video writer released.")