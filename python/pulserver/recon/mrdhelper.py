# MRD Helper functions
import base64
import re

import ismrmrd


def update_img_header_from_raw(imgHead, rawHead):
    """Populate ImageHeader fields from AcquisitionHeader"""

    if rawHead is None:
        return imgHead

    # # These fields are not translated from the raw header, but filled in
    # # during image creation by from_array
    # imgHead.data_type            =
    # imgHead.matrix_size          =
    # imgHead.channels             =

    # # This is mandatory, but must be filled in from the XML header,
    # # not from the acquisition header
    # imgHead.field_of_view        =

    imgHead.version = rawHead.version
    imgHead.flags = rawHead.flags
    imgHead.measurement_uid = rawHead.measurement_uid

    imgHead.position = rawHead.position
    imgHead.read_dir = rawHead.read_dir
    imgHead.phase_dir = rawHead.phase_dir
    imgHead.slice_dir = rawHead.slice_dir
    imgHead.patient_table_position = rawHead.patient_table_position

    imgHead.average = rawHead.idx.average
    imgHead.slice = rawHead.idx.slice
    imgHead.contrast = rawHead.idx.contrast
    imgHead.phase = rawHead.idx.phase
    imgHead.repetition = rawHead.idx.repetition
    imgHead.set = rawHead.idx.set

    imgHead.acquisition_time_stamp = rawHead.acquisition_time_stamp
    imgHead.physiology_time_stamp = rawHead.physiology_time_stamp

    # Defaults, to be updated by the user
    imgHead.image_type = ismrmrd.IMTYPE_MAGNITUDE
    imgHead.image_index = 1
    imgHead.image_series_index = 0

    imgHead.user_float = rawHead.user_float
    imgHead.user_int = rawHead.user_int

    return imgHead


def get_userParameterLong_value(metadata, name):
    """Get a value from MRD Header userParameterLong (returns None if key not found)"""
    if metadata.userParameters is not None:
        for param in metadata.userParameters.userParameterLong:
            if param.name == name:
                return int(param.value)
    return None


def get_userParameterDouble_value(metadata, name):
    """Get a value from MRD Header userParameterDouble (returns None if key not found)"""
    if metadata.userParameters is not None:
        for param in metadata.userParameters.userParameterDouble:
            if param.name == name:
                return float(param.value)
    return None


def get_userParameterString_value(metadata, name):
    """Get a value from MRD Header userParameterString (returns None if key not found)"""
    if metadata.userParameters is not None:
        for param in metadata.userParameters.userParameterString:
            if param.name == name:
                return param.value
    return None


def get_userParameterBase64_value(metadata, name):
    """Get a value from MRD Header userParameterBase64 (returns None if key not found)"""
    if metadata.userParameters is not None:
        for param in metadata.userParameters.userParameterBase64:
            if param.name == name:
                return base64.b64decode(param.value).decode("utf-8")
    return None


def get_sequence_resource_paths(metadata):
    """Return the pulserver sequence-resource paths from the MRD XMLHeader.

    These are written by the VRE OnPrep stage of the pulserver-interpreter
    (see ``mrdserver/csrc/trajectory_cache_reader.cpp::add_sequence_resource_paths``)
    based on the rhuser6/7 raw-header slots set by the PSD:

    * ``tensor_dat_path``: absolute path to a diffusion tensor file
      (e.g. ``/usr/g/bin/tensor7.dat``). Present only when the PSD's
      ``opuser3`` is non-zero.
    * ``grad_coef_path``: absolute path to a gradient coefficient file
      matching the scanner's gradient subsystem (``cfgradcoil``).

    Either entry may be missing; the caller is responsible for handling
    ``None`` values.

    Returns
    -------
    dict with keys ``"tensor_dat_path"`` and ``"grad_coef_path"``. Values
    are ``str`` if the entry is present, otherwise ``None``.
    """
    return {
        "tensor_dat_path": get_userParameterString_value(metadata, "tensor_dat_path"),
        "grad_coef_path": get_userParameterString_value(metadata, "grad_coef_path"),
    }


def get_meta_value(meta, key):
    """Get a value from MRD Meta Attributes (returns None if key not found)"""
    if key in meta:
        return meta[key]
    else:
        return None


def extract_minihead_bool_param(miniHead, name):
    """Extract a bool parameter from the serialized text of the ICE MiniHeader"""
    val = extract_minihead_param(miniHead, name, "ParamBool")

    if val is None:
        return False
    return val.strip('" ').lower() == "true".lower()


def extract_minihead_long_param(miniHead, name):
    """Extract a long parameter from the serialized text of the ICE MiniHeader"""
    val = extract_minihead_param(miniHead, name, "ParamLong")

    if val is None:
        return 0
    else:
        return int(val)


def extract_minihead_double_param(miniHead, name):
    """Extract a double parameter from the serialized text of the ICE MiniHeader"""
    val = extract_minihead_param(miniHead, name, "ParamDouble")

    if val is None:
        return float(0)
    else:
        return float(val)


def extract_minihead_string_param(miniHead, name):
    """Extract a string parameter from the serialized text of the ICE MiniHeader"""
    val = extract_minihead_param(miniHead, name, "ParamString")

    return val.strip(' "')


def extract_minihead_param(miniHead, name, strType):
    """Extract a string parameter from the serialized text of the ICE MiniHeader"""
    expr = r"(?<=<" + strType + r'."' + name + r'">)\s*[^}]*\s*'
    res = re.search(expr, miniHead)

    if res is None:
        return None

    # Strip off beginning '{' and whitespace, then split on newlines
    values = res.group(0).strip("{\n ").split("\n")

    # Lines beginning with <> are properties -- ignore them
    values = [val for val in values if bool(re.search(r"^\s*<\w+>", val)) is False]

    if len(values) != 1:
        return None
    else:
        return values[0]


def get_json_config_param(config, key, default=None, type="str"):
    """
    Read a parameter from JSON config
        Input:
            - config  : dict of parameters
            - key     : name (key) of parameter
            - default : value if key is not present or config is invalid
            - type    : type casting of the parameter (int, float, string, bool)
        Output:
            - value of parameter, or default if absent
    """
    if not isinstance(config, dict):
        return default

    if "parameters" not in config:
        return default

    if key not in config["parameters"]:
        return default

    value = config["parameters"][key]

    if type == "int":
        return int(value)
    elif (type == "float") or (type == "double"):
        return float(value)
    elif (type == "string") or (type == "str") or (type == "choice"):
        return str(value)
    elif (type == "bool") or (type == "boolean"):
        if isinstance(value, bool):
            return value
        elif "true" in value.lower():
            return True
        elif "false" in value.lower():
            return False
        else:
            return default
    else:
        raise Exception("'type' must be int, float, string, or bool")


def create_roi(x, y, rgb=(1, 0, 0), thickness=1, style: int = 0, visibility: int = 1):
    """
    Create an MRD-formatted ROI
        Parameters:
            - x (1D ndarray)     : x coordinates in units of pixels, with (0,0) at the top left
            - y (1D ndarray)     : y coordinates in units of pixels, matching the length of x
            - rgb (3 item tuple) : Colour as an (red, green, blue) tuple normalized to 1
            - thickness (float)  : Line thickness
            - style (int)        : Line style (0 = solid, 1 = dashed)
            - visibility (int)   : Line visibility (0 = false, 1 = true)
        Returns:
            - roi (string list)  : MRD-formatted ROI, intended to be stored as a MetaAttribute
                                   with field name starting with "ROI_"
    """
    xy = [(x[i], y[i]) for i in range(0, len(x))]  # List of (x,y) tuples

    roi = []
    roi.append(f"{rgb[0]:f}")
    roi.append(f"{rgb[1]:f}")
    roi.append(f"{rgb[2]:f}")
    roi.append(f"{thickness:f}")
    roi.append(f"{style:d}")
    roi.append(f"{visibility:d}")

    for i in range(0, len(xy)):
        roi.append(f"{xy[i][0]:f}")
        roi.append(f"{xy[i][1]:f}")

    return roi


def parse_roi(roi):
    """
    Parse an MRD-formatted ROI
        Input:
            - roi (string list)  : MRD-formatted ROI from a MetaAttribute
        Output:
            - x (1D ndarray)     : x coordinates in units of pixels, with (0,0) at the top left
            - y (1D ndarray)     : y coordinates in units of pixels, matching the length of x
            - rgb (3 item tuple) : Colour as an (red, green, blue) tuple normalized to 1
            - thickness (float)  : Line thickness
            - style (int)        : Line style (0 = solid, 1 = dashed)
            - visibility (int)   : Line visibility (0 = false, 1 = true)
    """
    if (not isinstance(roi, list)) or (len(roi) < 8) or (len(roi) % 2):
        raise Exception(
            "ROI must be a list, have 6 metadata values, at least one coordinate, and an even number of values (x,y pairs)"
        )

    fRoi = [float(x) for x in roi]

    rgb = tuple(fRoi[0:3])
    thickness = fRoi[3]
    style = int(fRoi[4])
    visibility = int(fRoi[5])

    x = fRoi[6::2]
    y = fRoi[7::2]

    return x, y, rgb, thickness, style, visibility


def create_text(x, y, rgb=(1, 0, 0), visibility: int = 1, string=""):
    """
    Create an MRD-formatted text object
        Parameters:
            - x (float)          : x coordinate in units of pixels, with (0,0) at the top left
            - y (float)          : y coordinate in units of pixels
            - rgb (3 item tuple) : Colour as an (red, green, blue) tuple normalized to 1
            - visibility (int)   : Line visibility (0 = false, 1 = true)
            - string (string)    : Text string
        Returns:
            - txt (string list)  : MRD-formatted text, intended to be stored as a MetaAttribute
                                   with field name starting with "Text_"
    """
    txt = []
    txt.append(f"{rgb[0]:f}")
    txt.append(f"{rgb[1]:f}")
    txt.append(f"{rgb[2]:f}")
    txt.append(f"{x:f}")
    txt.append(f"{y:f}")
    txt.append(f"{visibility:d}")
    txt.append(f"{string:s}")

    return txt


def parse_text(txt):
    """
    Parse an MRD-formatted text object
        Input:
            - txt (string list)  : MRD-formatted text from a MetaAttribute
        Output:
            - x (float)          : x coordinate in units of pixels, with (0,0) at the top left
            - y (float)          : y coordinate in units of pixels
            - rgb (3 item tuple) : Colour as an (red, green, blue) tuple normalized to 1
            - visibility (int)   : Line visibility (0 = false, 1 = true)
            - string (string)    : Text string
    """
    if (not isinstance(txt, list)) or (len(txt) != 7):
        raise Exception("txt must be a list that has exactly 7 metadata values")

    rgb = tuple([float(x) for x in txt[0:3]])
    x = float(txt[3])
    y = float(txt[4])
    visibility = int(float(txt[5]))

    string = txt[6]

    return x, y, rgb, visibility, string
