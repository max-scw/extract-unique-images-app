from typing import Dict


def create_auth_headers(token: str = None) -> Dict[str, str] | None:
    if token:
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/octet-stream'
        }
    else:
        headers = None
    return headers