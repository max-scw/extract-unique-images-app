import requests
import urllib
from timeit import default_timer

from PIL import Image
from typing import Union, Dict, List, Any, Optional


from DataModels_BaslerCameraAdapter import BaslerCameraSettings, ImageParams
from utils import create_auth_headers, setup_logging
from utils_image import bytes_to_image_pil


# Setup logging
logger = setup_logging(__name__)


def build_url(
        address: str,
        camera_params: BaslerCameraSettings,
        image_params: ImageParams,
) -> str:
    """
    Builds the url to call the BaslerCameraAdapter container
    gitHub project: https://github.com/max-scw/BaslerCameraAdapter
    container: https://hub.docker.com/r/maxscw/baslercameraadapter
    :param address: url to REST api of the Basler camera
    :param camera_params:
    :param image_params:
    :return: formatted url
    """

    logger.debug(f"build_url({address}, {camera_params}, {image_params})")
    # address
    if not address.startswith(("http://", "https://")):
        address = "http://" + address

    # join dictionaries
    parameter = camera_params.model_dump() | image_params.model_dump()

    # parameter
    params = {ky: vl for ky, vl in parameter.items() if (vl not in (None, "", "token"))}

    info = {ky: (vl, type(vl)) for ky, vl in params.items()}
    logger.debug(f"building URL for backend: {info}")
    # build url
    return f"{address}?{urllib.parse.urlencode(params)}"




def request_image_from_camera(
        address: str,
        camera_params: BaslerCameraSettings,
        image_params: ImageParams,
        timeout: int = 5,  # seconds
        token = Optional[str]
) -> Union[Image.Image, None]:
    """
    wrapper
    """
    # synchronize timeout between request and pylon.RetrieveImage function
    camera_params.timeout_ms = min([camera_params.timeout_ms, int(timeout * 1000)])

    t0 = default_timer()
    url = build_url(address, camera_params, image_params)

    content = request_camera(url, timeout, token=token)
    t1 = default_timer()
    logger.debug(f"trigger_camera(): request_camera(url={url}, timeout={timeout}) (took {(t1 - t0) * 1000:.4g} ms)")

    if content is not None:
        content = bytes_to_image_pil(content)
    return content


def request_camera(
        address: str,
        timeout: int = 5,  # seconds
        token: str = None
) -> Union[bytes, None]:

    t0 = default_timer()
    response = requests.get(url=address, timeout=timeout, headers=create_auth_headers(token))
    status_code = response.status_code
    t1 = default_timer()
    logger.info(
        f"Requesting camera {address} took {(t1 - t0) * 1000:.4g} ms. "
        f"(Status code: {status_code})"
    )

    # status codes
    # 1xx: Informational – Communicates transfer protocol-level information.
    # 2xx: Success – Indicates that the client’s request was accepted successfully.
    # 3xx: Redirection – Indicates that the client must take some additional action in order to complete their request.
    # 4xx: Client Error – This category of error status codes points the finger at clients.
    # 5xx: Server Error – The server takes responsibility for these error status codes.
    content = None
    if 200 <= status_code < 300:
        # output buffer
        content = response.content
    elif 400 <= status_code < 600:
        # error
        raise Exception(f"Server returned status code {status_code} with message {response.text}")

    return content

