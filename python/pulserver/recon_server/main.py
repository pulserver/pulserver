"""CLI entrypoint for pulserver minimal recon server."""

from __future__ import annotations

import argparse
import logging

from .server import Server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Pulserver minimal recon server')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=9002)
    parser.add_argument('--default-plugin', default='simplefft')
    parser.add_argument('--plugin-directory', default='')
    parser.add_argument('--savedata', action='store_true')
    parser.add_argument('--savedata-folder', default='/tmp/share/saved_data')
    parser.add_argument('--multiprocessing', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(format='%(asctime)s - %(message)s', level=level)

    server = Server(
        args.host,
        args.port,
        default_plugin=args.default_plugin,
        savedata=args.savedata,
        savedata_folder=args.savedata_folder,
        use_multiprocessing=args.multiprocessing,
        plugin_directory=args.plugin_directory or None,
    )
    server.serve()


if __name__ == '__main__':
    main()
