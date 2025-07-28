from redis._parsers.helpers import pairs_to_dict as pairs_to_dict
from redis.commands.vectorset.commands import CallbacksOptions as CallbacksOptions

def parse_vemb_result(response, **options):
    """
    Handle VEMB result since the command can returning different result
    structures depending on input options and on quantization type of the vector set.

    Parsing VEMB result into:
    - List[Union[bytes, Union[int, float]]]
    - Dict[str, Union[bytes, str, float]]
    """
def parse_vlinks_result(response, **options):
    """
    Handle VLINKS result since the command can be returning different result
    structures depending on input options.
    Parsing VLINKS result into:
    - List[List[str]]
    - List[Dict[str, Number]]
    """
def parse_vsim_result(response, **options):
    """
    Handle VSIM result since the command can be returning different result
    structures depending on input options.
    Parsing VSIM result into:
    - List[List[str]]
    - List[Dict[str, Number]]
    """
