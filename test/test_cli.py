import pytest
import argparse
from src.cli import create_argument_parser, parse_and_validate_args


def test_create_argument_parser():
    """Test CLI argument parser creation."""
    parser = create_argument_parser()

    assert isinstance(parser, argparse.ArgumentParser)

    # Test default scenario
    args = parser.parse_args([])
    assert args.scenario == "scenarios/default.yaml"
    assert args.output is None
    assert args.no_charts is False


def test_parse_scenario_argument():
    """Test parsing scenario file argument."""
    parser = create_argument_parser()

    args = parser.parse_args(["--scenario", "custom.yaml"])
    assert args.scenario == "custom.yaml"


def test_parse_output_directory():
    """Test parsing output directory argument."""
    parser = create_argument_parser()

    args = parser.parse_args(["--output", "results/"])
    assert args.output == "results/"


def test_parse_no_charts_flag():
    """Test parsing no-charts flag."""
    parser = create_argument_parser()

    args = parser.parse_args(["--no-charts"])
    assert args.no_charts is True


def test_parse_verbose_flag():
    """Test parsing verbose flag."""
    parser = create_argument_parser()

    args = parser.parse_args(["-v"])
    assert args.verbose is True
