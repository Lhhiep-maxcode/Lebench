"""Small HTTP helpers for explicitly configured internal API endpoints."""

import urllib.request
from urllib.parse import urlsplit


def api_base_url(value):
    """Reject non-HTTP schemes, embedded credentials and ambiguous API URLs."""
    if not isinstance(value, str) or not value:
        raise ValueError('API base URL must be an HTTP or HTTPS URL')
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise ValueError('Invalid API base URL') from exc
    if (parsed.scheme not in ('http', 'https') or not parsed.hostname or
            parsed.username is not None or parsed.password is not None or
            parsed.query or parsed.fragment or port == 0):
        raise ValueError('API base URL must be HTTP(S), without credentials, query or fragment')
    return value.rstrip('/')


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


def urlopen_direct(request, timeout):
    """Do not use ambient proxies or follow redirects to another endpoint."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect)
    return opener.open(request, timeout=timeout)
