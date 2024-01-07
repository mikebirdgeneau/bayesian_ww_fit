import argparse
import os
import traceback
import warnings

import requests

from dotenv import load_dotenv

from wastewater import WastewaterModel

# Load .env file
load_dotenv()

def notify(text, title="Bayesian Wastewater Process", priority='default', tags=None):
    """Send a notification to ntfy.sh"""
    ntfy_endpoint = os.getenv('NTFY_ENDPOINT', "https://ntfy.sh")
    ntfy_channel = os.getenv('NTFY_CHANNEL', "3ea72770b3c56176769740a36c6a1d90")

    headers = {
        'Title': title,
    }
    if priority is not None:
        headers['Priority'] = priority

    if tags is not None:
        headers['Tags'] = ','.join(tags)

    
    requests.post("{}/{}".format(ntfy_endpoint,ntfy_channel),
                  data=text,
                  headers=headers)
    pass


def error_hook(func):
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            tb_str = traceback.format_exc()
            error_msg = f"Error: {e}\nTraceback:\n{tb_str}"
            notify(error_msg, title="Error in Bayesian Wastewater Process", priority='high', tags=['warning', 'skull'])
            raise e
    return wrapper


@error_hook
def main():
    # Take location from command line argument:
    parser = argparse.ArgumentParser()
    parser.add_argument('--location', type=str, required=True, help="Location name (e.g. Calgary, Edmonton, etc.)")
    args = parser.parse_args()
    location = args.location

    # if location argument is missing, display help:
    if location is None:
        parser.print_help()
        exit(1)

    # Define model:
    notify("Starting Bayesian Wastewater Process for {}".format(location),tags=['green_circle','hourglass'])
    ww = WastewaterModel(location=location, exclude_last_n_days=0)

    # Perform HMC sampling:
    ww.fit_model()

    # Plot traces (for diagnostics / debugging):
    ww.plot_traces()
    ww.plot_hist()

    # Forecast from last sample to today:
    ww.predict()

    # Plot one-step ahead forecast:
    ww.predict_one_step()

    # Plot forecast:
    ww.plot_forecast()

    # Export CSV:
    ww.export_csv()

    # Components:
    ww.component_distributions()

    # Notify:
    requests.put("{}/{}".format(
        os.getenv('NTFY_ENDPOINT', "https://ntfy.sh"),
        os.getenv("NTFY_CHANNEL","3ea72770b3c56176769740a36c6a1d90")),
        data=open("output/forecast_{}.png".format(location), 'rb'),
        headers={
            "Title": "Wastewater Trend for {}".format(location),
            "Tags": "checkered_flag",
            "Filename": "forecast_{}.png".format(location)})

if __name__ == "__main__":
    main()
