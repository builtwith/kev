"""Download and cache shared question documents without touching disk."""
from collections import OrderedDict
from copy import deepcopy
import json
import logging
import os
import threading
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener
from urllib.error import URLError

from fastapi import HTTPException
from pydantic import TypeAdapter, ValidationError
from .api import Question

MAX_BYTES = 8 * 1024 * 1024
CACHE_SIZE = 16
ALLOWED_HOSTS = frozenset(h.strip().lower() for h in os.environ.get(
    "KEV_QUESTION_URL_HOSTS", "").split(",") if h.strip())
_cache = OrderedDict()
_lock = threading.Lock()
_adapter = TypeAdapter(dict[str, Question])
_log = logging.getLogger(__name__)


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def resolve_questions(value):
    if not isinstance(value, str):
        return value
    try:
        url = urlsplit(value)
        valid = (url.scheme == "https" and url.hostname in ALLOWED_HOSTS
                 and url.port in (None, 443) and not url.username and not url.password
                 and not url.fragment)
    except ValueError:
        valid = False
    if not valid:
        raise HTTPException(422, "questions URL must use HTTPS on an allowed host (KEV_QUESTION_URL_HOSTS), without credentials or fragment")
    # Serialize cold downloads so concurrent requests fetch a URL only once.
    # This lock is separate from the model lock; endpoints run in FastAPI's thread pool.
    with _lock:
        if value in _cache:
            _cache.move_to_end(value)
            return deepcopy(_cache[value])
        try:
            opener = build_opener(ProxyHandler({}), NoRedirects())
            with opener.open(Request(value, headers={"Accept": "application/json"}), timeout=30) as response:
                raw = response.read(MAX_BYTES + 1)
            if len(raw) > MAX_BYTES:
                raise HTTPException(413, "questions document exceeds 8 MiB")
            questions = _adapter.validate_python(json.loads(raw))
            if not questions:
                raise ValueError("empty questions")
        except (URLError, OSError):
            raise HTTPException(502, "Unable to download questions document") from None
        except (ValueError, ValidationError):
            raise HTTPException(422, "Downloaded questions document is not a valid nonempty question object") from None
        _cache[value] = questions
        while len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)
        _log.info("Cached questions document (%d bytes, %d questions)", len(raw), len(questions))
        return deepcopy(questions)
