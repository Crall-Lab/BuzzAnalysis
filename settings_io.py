"""Shared settings-file validation and explicit command-line precedence."""
import json
from pathlib import Path
import sys


def read_settings(path):
    path = Path(path).expanduser()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Settings must be a JSON object: {path}")
    return data


def parse_with_overrides(parser, argv=None):
    """Remember explicitly supplied options so JSON cannot replace CLI choices."""
    parser.allow_abbrev = False
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv)
    options = {option: action.dest for action in parser._actions for option in action.option_strings}
    args._explicit_options = {options[token.split('=', 1)[0]] for token in argv
                              if token.split('=', 1)[0] in options}
    return args
