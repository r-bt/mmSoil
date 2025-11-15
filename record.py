import argparse
import numpy as np
from datetime import datetime
import os

from src.iwr1443.radar import Radar

buffer = []


def log(msg):
    """
    Callback function to log the data to the csv file
    """
    global buffer

    buffer.append(msg["data"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record data from the DCA1000")
    parser.add_argument("--cfg", type=str, required=True, help="Path to the .lua file")

    args = parser.parse_args()

    # Initalize the npz writer
    filename = f"data/radar_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
    os.makedirs("data", exist_ok=True)

    # Initialize the radar
    radar = Radar(args.cfg, host_ip="192.168.33.42")
    radar.run_polling(cb=log)

    print("Saving data...")

    # Save the data
    np.savez_compressed(filename, data=np.stack(buffer))

    print(f"Saved data to {filename}")
