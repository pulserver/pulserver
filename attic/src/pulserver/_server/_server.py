"""Main sequence design server app."""

__all__ = ["start_server", "load_plugins", "load_config"]

import configparser
import importlib.util
import logging
import os
import pathlib
import socket

from datetime import datetime

from ..parsing import ParamsParser

# Location of home
HOME_DIR = pathlib.Path.home()

# Define the default config file location
DEFAULT_CONFIG_PATH = os.path.join(HOME_DIR, ".pulserver_config.ini")
DEFAULT_LOGDIR = os.path.join(HOME_DIR, "pulserver_log")
DEFAULT_PLUGINDIR = os.path.join(HOME_DIR, "pulserver_plugins")

# Default configuration values
DEFAULT_CONFIG = {
    "SCANNER_ADDRESS": "127.0.0.1",
    "SCANNER_PORT": 5000,
    "RECON_SEVER_ADDRESS": None,
    "RECON_SEVER_PORT": None,
    "PLUGINSDIR": DEFAULT_PLUGINDIR,
    "LOGDIR": DEFAULT_LOGDIR,
}


# Load configuration
def load_config():
    """
    Load configuration from the pulserver.ini file.

    First, check the PULSERVER_CONFIG environment variable for the file location.
    If not set or file does not exist, fall back to the default config location.
    If no config file is found, use default values.
    Returns the configuration as a dictionary.


    """
    config = DEFAULT_CONFIG.copy()

    # Check the PULSECLIENT_CONFIG environment variable
    config_file = os.getenv("PULSERVER_CONFIG", DEFAULT_CONFIG_PATH)

    # If the file exists in the specified or default location, load the configuration
    if os.path.exists(config_file):
        print(
            "Loading configuration from: {}".format(config_file)
        )  # Use .format for compatibility
        parser = configparser.ConfigParser()
        parser.read(config_file)

        # Check if the "settings" section exists
        if parser.has_section("settings"):  # Use has_section instead of direct check
            config.update(
                {
                    "SCANNER_ADDRESS": parser.get(
                        "settings",
                        "SCANNER_ADDRESS",
                        fallback=config["SCANNER_ADDRESS"],
                    ),
                    "SCANNER_PORT": parser.getint(
                        "settings", "SCANNER_PORT", fallback=config["SCANNER_PORT"]
                    ),
                    "RECON_SEVER_ADDRESS": parser.get(
                        "settings",
                        "RECON_SEVER_ADDRESS",
                        fallback=config["RECON_SEVER_ADDRESS"],
                    ),
                    "RECON_SEVER_PORT": parser.getint(
                        "settings",
                        "RECON_SEVER_PORT",
                        fallback=config["RECON_SEVER_PORT"],
                    ),
                    "RECON_SEVER_USER": parser.get(
                        "settings",
                        "RECON_SEVER_USER",
                        fallback=config["RECON_SEVER_USER"],
                    ),
                    "RECON_SEVER_HOST": parser.get(
                        "settings",
                        "RECON_SEVER_HOST",
                        fallback=config["RECON_SEVER_HOST"],
                    ),
                    "RECON_SERVER_COMMAND": parser.get(
                        "settings",
                        "RECON_SERVER_COMMAND",
                        fallback=config["RECON_SERVER_COMMAND"],
                    ),
                    "RECON_SERVER_PROCESS_NAME": parser.get(
                        "settings",
                        "RECON_SERVER_PROCESS_NAME",
                        fallback=config["RECON_SERVER_PROCESS_NAME"],
                    ),
                    "PLUGINSDIR": parser.get(
                        "settings",
                        "PLUGINSDIR",
                        fallback=config["PLUGINSDIR"],
                    ),
                    "LOGDIR": parser.get(
                        "settings",
                        "LOGDIR",
                        fallback=config["LOGDIR"],
                    ),
                }
            )
    else:
        print("No config file found at {}. Using default values.".format(config_file))

    return config


# Plugins
def _get_plugin_dir(config):
    # Read built-int apps
    PKG_DIR = pathlib.Path(os.path.realpath(__file__)).parents[1].resolve()
    PLUGIN_DIR = [os.path.join(PKG_DIR, "_apps")]

    # Add custom design functions
    CUSTOM_PLUGINS = config["PLUGINSDIR"]
    if CUSTOM_PLUGINS and os.path.exists(CUSTOM_PLUGINS):
        PLUGIN_DIR.append(os.path.realpath(CUSTOM_PLUGINS))

    return PLUGIN_DIR


# Logs
def _get_log_dir(config):
    # Get environment variable
    LOG_DIR = config["LOGDIR"]
    if LOG_DIR is None:  # Default to user HOME folder
        LOG_DIR = DEFAULT_LOGDIR

    # Ensure log directory exists
    os.makedirs(LOG_DIR, exist_ok=True)

    return LOG_DIR


# Configure main session logging
def setup_main_logger(config):
    LOG_DIR = _get_log_dir(config)
    session_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_log_filename = os.path.join(LOG_DIR, f"session_{session_start_time}.log")
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(main_log_filename), logging.StreamHandler()],
    )
    logger = logging.getLogger("main")
    return logger


def setup_function_logger(config, function_name):
    LOG_DIR = _get_log_dir(config)
    function_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    function_log_filename = os.path.join(
        LOG_DIR, f"{function_name}_{function_start_time}.log"
    )
    function_logger = logging.getLogger(function_name)
    function_logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(function_log_filename)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    function_logger.addHandler(handler)
    return function_logger


def load_plugins(config, logger=None):  # noqa
    # Get plugin path
    PLUGIN_DIR = _get_plugin_dir(config)

    # Load plugins
    plugins = {}
    for directory in PLUGIN_DIR:
        for filename in os.listdir(directory):
            if filename.endswith(".py"):
                filepath = os.path.join(directory, filename)
                module_name = filename[:-3]
                func_name = module_name
                # do the import
                spec = importlib.util.spec_from_file_location(module_name, filepath)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                try:
                    func = getattr(module, func_name)
                except Exception:
                    logger.error(
                        f"Plugin function  {func_name} must have the same name as its module {module_name}."
                    )
                    raise ImportError
                plugins[module_name] = func
                if logger is not None:
                    logger.debug(f"Loaded plugin: {module_name} from {filepath}")
    return plugins


def parse_request(request, logger):
    try:
        # Example format: "funcname n var1 var2 ... varn"
        params = ParamsParser.from_bytes(request)
        function_name = params.function_name
        kwargs = params.asdict()
        logger.debug(
            f"Parsed request - Function: {function_name}, Keyworded Args: {kwargs}"
        )
        return function_name, kwargs
    except Exception as e:
        logger.error(f"Failed to parse request: {e}")
        return None, None


def send_to_recon_server(optional_buffer, config):
    RECON_SERVER_ADDRESS = config.get("RECON_SERVER_ADDRESS", None)
    RECON_SERVER_PORT = config.get("RECON_SERVER_PORT", None)
    if RECON_SERVER_ADDRESS is not None and RECON_SERVER_PORT is not None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((RECON_SERVER_ADDRESS, RECON_SERVER_PORT))
            s.sendall(optional_buffer)
            s.shutdown(socket.SHUT_WR)


def handle_client_connection(config, client_socket, plugins, logger):
    request = client_socket.recv(1024)
    function_name, kwargs = parse_request(request, logger)
    if function_name in plugins:
        # Select function
        function = plugins[function_name]

        # Set-up logging
        function_logger = setup_function_logger(config, function_name)
        logger.info(f"Calling {function_name} with args {kwargs}")

        # Run design function
        result_buffer, optional_buffer = function(kwargs)

        # Log the output to the function-specific log file
        function_logger.info(f"Output buffer: {result_buffer}")

        # Send the result buffer to the client
        client_socket.sendall(result_buffer)

        # Signal that no more data will be sent
        client_socket.shutdown(socket.SHUT_WR)

        # Optionally send the reconstruction info to the secondary server
        if optional_buffer is not None:
            send_to_recon_server(optional_buffer, config)
    else:
        logger.error(f"Function {function_name} not found")


def start_server(config):  # noqa
    SCANNER_ADDRESS = config["SCANNER_ADDRESS"]
    SCANNER_PORT = config["SCANNER_PORT"]

    # Set-up main logger
    logger = setup_main_logger(config)

    # Load plugins
    plugins = load_plugins(config, logger)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((SCANNER_ADDRESS, SCANNER_PORT))
        s.listen()
        logger.info(f"Server listening on {SCANNER_ADDRESS}:{SCANNER_PORT}")
        while True:
            conn, addr = s.accept()
            with conn:
                logger.info(f"Connected by {addr}")
                handle_client_connection(config, conn, plugins, logger)
