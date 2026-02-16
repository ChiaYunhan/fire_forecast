"""Command-line interface for FIRE Forecast Engine."""
import argparse
import sys
from pathlib import Path


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser for CLI.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="FIRE Forecast Engine - Monte Carlo retirement projections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default scenario
  %(prog)s

  # Run with custom scenario
  %(prog)s --scenario my_scenario.yaml

  # Save charts to directory
  %(prog)s --output results/

  # Skip chart generation (faster)
  %(prog)s --no-charts
        """
    )

    parser.add_argument(
        "-s", "--scenario",
        default="scenarios/default.yaml",
        help="Path to scenario YAML file (default: scenarios/default.yaml)"
    )

    parser.add_argument(
        "-o", "--output",
        help="Output directory for saving charts (optional)"
    )

    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Skip chart generation (print results only)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    return parser


def parse_and_validate_args(args=None) -> argparse.Namespace:
    """Parse and validate command-line arguments.

    Args:
        args: List of argument strings (None = sys.argv)

    Returns:
        Parsed arguments namespace

    Raises:
        SystemExit: If validation fails
    """
    parser = create_argument_parser()
    parsed = parser.parse_args(args)

    # Validate scenario file exists
    scenario_path = Path(parsed.scenario)
    if not scenario_path.exists():
        parser.error(f"Scenario file not found: {parsed.scenario}")

    # Validate output directory if specified
    if parsed.output:
        output_path = Path(parsed.output)
        if output_path.exists() and not output_path.is_dir():
            parser.error(f"Output path exists but is not a directory: {parsed.output}")

    return parsed
