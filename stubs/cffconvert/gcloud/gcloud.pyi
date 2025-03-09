from cffconvert import Citation as Citation
from cffconvert.cli.read_from_url import read_from_url as read_from_url

def get_help_text(): ...
def cffconvert(request):
    """Responds to any HTTP request.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using
        `make_response <http://flask.pocoo.org/docs/1.0/api/#flask.Flask.make_response>`.
    """
