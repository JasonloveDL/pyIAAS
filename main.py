import click

from pyIAAS import run_predict, run_search


@click.group()
def cli():
    click.echo('Welcome to pyIAAS!!!')


@cli.command()
@click.option('-c', help='Configuration file path')
@click.option('-f', help='Input CSV file')
@click.option('-t', help='Target value name in CSV file')
@click.option('-r', default=0.2, help='ratio of test dataset')
def search(c, f, t, r):
    config_file = c
    input_file = f
    target_name = t
    test_ratio = r
    run_search(config_file, input_file, target_name, test_ratio)


@cli.command()
@click.option('-c', help='Configuration file path')
@click.option('-d', help='output directory of previous search result')
@click.option('-f', help='Input CSV file, file should contains more than predict length row of data')
@click.option('-t', help='Target value name in CSV file')
@click.option('-o', help='Output prediction file')
def predict(c, d, f, t, o):
    config_file = c
    input_file = f
    target_name = t
    output_dir = d
    prediction_file = o
    run_predict(config_file, input_file, target_name, output_dir, prediction_file)


if __name__ == '__main__':
    cli()
