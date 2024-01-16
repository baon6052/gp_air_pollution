import csv

import click

from gaussian_process import run_model


@click.command()
@click.option("--num_samples", default=5)
def main(num_samples: int):
    mae, _, _ = run_model(
        climate_variables=[], num_samples=num_samples, plotting=False
    )
    with open("exp_data/num_start_samples.csv", "a+") as f:
        fieldnames = ["num_start_samples", "mae"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writerow(
            {
                "num_start_samples": num_samples,
                "mae": mae,
            }
        )


if __name__ == "__main__":
    main()
