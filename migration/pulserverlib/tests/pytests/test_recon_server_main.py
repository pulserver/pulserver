"""Unit tests for CLI parser defaults."""

from pulserver.recon_server.main import build_parser


def test_parser_defaults():
    parser = build_parser()
    args = parser.parse_args([])
    assert args.host == '0.0.0.0'
    assert args.port == 9002
    assert args.default_plugin == 'simplefft'
    assert args.savedata is False
