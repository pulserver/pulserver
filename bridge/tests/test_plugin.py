"""Minimal test plugin — no external dependencies required."""


def get_default_protocol(opts):
    return {
        "TE": {
            "type": "float", "value": 5.0, "min": 1.0, "max": 100.0,
            "incr": 0.1, "unit": "ms", "validate": "search",
        },
        "TR": {
            "type": "float", "value": 10.0, "min": 1.0, "max": 1000.0,
            "incr": 0.1, "unit": "ms", "validate": "search",
        },
        "FatSat": {
            "type": "bool", "value": False, "validate": "none",
        },
        "D_Header": {
            "type": "description", "text": "Minimal Test",
        },
    }


def validate_protocol(opts, protocol):
    te = protocol["TE"]["value"]
    tr = protocol["TR"]["value"]
    if te >= tr:
        return {"valid": False, "duration": None, "info": "TE must be < TR"}
    duration = tr * 256 / 1000.0
    return {"valid": True, "duration": duration, "info": f"TA = {duration:.2f} s"}


def make_sequence(opts, protocol, output_path):
    with open(output_path, "w") as f:
        f.write("# Minimal .seq file\n[VERSION] 1.4.0\n")
