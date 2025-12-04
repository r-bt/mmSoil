import argparse
import numpy as np
from datetime import datetime
import os
import threading
from queue import Queue

from src.iwr1443.radar import Radar

buffer = []
log_queue = Queue()
logging_active = threading.Event()
logging_active.set()


def log(msg):
    """
    Callback function to quickly enqueue data without blocking packet reception
    """
    # Non-blocking queue put - just adds to queue and returns immediately
    log_queue.put(msg["data"])


def logging_thread():
    """
    Background thread that processes the queue and appends to buffer
    """
    global buffer
    
    while logging_active.is_set() or not log_queue.empty():
        try:
            # Get data from queue with timeout to allow checking the active flag
            data = log_queue.get(timeout=0.1)
            buffer.append(data)
            log_queue.task_done()
        except:
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record data from the DCA1000")
    parser.add_argument("--cfg", type=str, required=True, help="Path to the .lua file")

    args = parser.parse_args()

    # Initalize the npz writer
    filename = f"data/radar_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
    os.makedirs("data", exist_ok=True)

    # Start the background logging thread
    logger = threading.Thread(target=logging_thread, daemon=True)
    logger.start()
    print("[INFO] Logging thread started")

    try:
        # Initialize the radar
        radar = Radar(args.cfg)
        radar.run_polling(cb=log)
    finally:
        # Signal the logging thread to stop and wait for it to finish
        print("\n[INFO] Stopping data capture...")
        logging_active.clear()
        
        # Wait for the queue to be fully processed
        print("[INFO] Processing remaining queued data...")
        log_queue.join()
        logger.join(timeout=5.0)
        
        print(f"[INFO] Saving {len(buffer)} frames...")

        # Save the data
        np.savez_compressed(filename, data=np.stack(buffer))

        print(f"[INFO] Saved data to {filename}")
