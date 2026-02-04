import time
import requests
from functools import wraps

from utils.logger import get_logger

logger = get_logger("API_GUARD")


class APIGuardException(Exception):
    pass


def safe_api_call(
    retries=3,
    timeout=10,
    backoff=2,
    allowed_exceptions=(requests.exceptions.RequestException,)
):
    """
    Decorator for safe API calls with retry + exponential backoff
    """

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            wait = 1

            while attempt < retries:
                try:
                    return func(*args, **kwargs)

                except allowed_exceptions as e:
                    attempt += 1
                    logger.warning(
                        f"API error on {func.__name__} attempt {attempt}/{retries} : {e}"
                    )

                    if attempt >= retries:
                        logger.error(f"API failed permanently: {func.__name__}")
                        raise APIGuardException(str(e))

                    time.sleep(wait)
                    wait *= backoff

                except Exception as e:
                    logger.error(f"Unexpected error in API call {func.__name__}: {e}")
                    raise

        return wrapper

    return decorator


# ============================
# SIMPLE SAFE REQUEST
# ============================

@safe_api_call()
def safe_get(url, params=None, headers=None, timeout=10):
    return requests.get(url, params=params, headers=headers, timeout=timeout)


@safe_api_call()
def safe_post(url, json=None, headers=None, timeout=10):
    return requests.post(url, json=json, headers=headers, timeout=timeout)
