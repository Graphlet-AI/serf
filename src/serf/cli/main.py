"""Main CLI entry point for SERF."""

import click

from serf.logs import get_logger

logger = get_logger(__name__)


@click.group(context_settings={"show_default": True})
@click.version_option()
def cli() -> None:
    """SERF: Semantic Entity Resolution Framework CLI."""
    pass


@cli.command()
@click.option(
    "--input",
    "-i",
    type=click.Path(exists=True),
    required=True,
    help="Input data file or directory",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    required=True,
    help="Output directory for results",
)
def block(input: str, output: str) -> None:
    """Perform semantic blocking on input data."""
    logger.info(f"Starting blocking with input: {input}, output: {output}")
    click.echo(f"Blocking data from {input} to {output}")
    # TODO: Implement blocking logic


@cli.command(name="match")
@click.option(
    "--input",
    "-i",
    type=click.Path(exists=True),
    required=True,
    help="Input directory with blocked data",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    required=True,
    help="Output directory for matched results",
)
def align_match_merge(input: str, output: str) -> None:
    """Align schemas, match entities, and merge within blocks."""
    logger.info(f"Starting align/match/merge with input: {input}, output: {output}")
    click.echo(f"Aligning, matching, and merging entities from {input} to {output}")
    # TODO: Implement align/match/merge logic


@cli.command(name="edges")
@click.option(
    "--input",
    "-i",
    type=click.Path(exists=True),
    required=True,
    help="Input directory with merged nodes",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    required=True,
    help="Output directory for resolved edges",
)
def edge_resolve(input: str, output: str) -> None:
    """Resolve edges after node merging."""
    logger.info(f"Starting edge resolution with input: {input}, output: {output}")
    click.echo(f"Resolving edges from {input} to {output}")
    # TODO: Implement edge resolution logic


if __name__ == "__main__":
    cli()
