import cv2
import os
import dask.array as da
import glob
import random
import datetime
import socket
import argparse
import sys
import logging
import time

# --- Configuration ---
DEFAULT_BASE_IMAGE_PATH = "/Volumes/googledrive_bombus2/ggm_fanning_2025/round1"
DEFAULT_NUM_IMAGES = 10

# --- Logging setup ---
log_dir = './logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'log.log'), encoding='utf-8',
                    format='%(filename)s %(asctime)s: %(message)s', filemode='a', level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# --- Utility functions ---
def find_latest_image_folder(base_path):
    all_subdirs = glob.glob(os.path.join(base_path, '*', '*'))  # e.g., round1/mc1_2/2025-07-03
    candidate_folders = []

    for path in all_subdirs:
        if glob.glob(os.path.join(path, '*.png')):
            candidate_folders.append(path)

    if not candidate_folders:
        raise FileNotFoundError("No folders with PNG images found.")

    candidate_folders.sort(reverse=True)
    return candidate_folders[0]

def extract_mc_folder(image_folder_path):
    parts = image_folder_path.strip(os.sep).split(os.sep)
    if len(parts) < 2:
        raise ValueError("Cannot extract mc folder from image folder path.")
    return os.path.join("/", *parts[:-1])  # up to mc1_2

def create_nest_images_folder(mc_folder):
    nest_folder = os.path.join(mc_folder, "Nest Images")
    os.makedirs(nest_folder, exist_ok=True)
    return nest_folder

def generate_nest_image(image_folder, number_of_images, hostname, date_str, output_folder, shuffle=True):
    image_paths = glob.glob(os.path.join(image_folder, '*.png'))
    if len(image_paths) == 0:
        logger.warning(f"No PNG images found in {image_folder}")
        return 0

    if shuffle:
        random.shuffle(image_paths)

    total_frames = min(len(image_paths), number_of_images)
    image_paths = image_paths[:total_frames]

    print(f"Generating composite image for {image_folder} using {total_frames} images")

    # Load first valid image to get dimensions
    first_img = None
    for p in image_paths:
        img = cv2.imread(p)
        if img is not None:
            first_img = img
            break

    if first_img is None:
        print(f"Error: None of the images in {image_folder} could be loaded.")
        return 0

    gray_img = cv2.cvtColor(first_img, cv2.COLOR_RGB2GRAY)
    h, w = gray_img.shape
    x = da.zeros((total_frames, h, w), dtype='uint8')

    valid_img_count = 0
    for idx, img_path in enumerate(image_paths):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not load image {img_path}, skipping.")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if gray.shape != (h, w):
            gray = cv2.resize(gray, (w, h))
        x[valid_img_count] = gray
        valid_img_count += 1

    if valid_img_count == 0:
        print(f"No valid images found to create composite in {image_folder}")
        return 0

    median_img = da.median(x[:valid_img_count], axis=0).compute()

    create_folder_if_needed(output_folder)

    filename = f"{hostname}-{date_str}-nest_image.png"
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, median_img)

    print(f"Saved composite image to: {output_path}")
    logger.info(f"Composite image created at {output_path} using {valid_img_count} images.")
    return valid_img_count


# --- Main ---
def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(description="Generate a nest image from a set of PNGs.")
    parser.add_argument('-b', '--base_path', type=str, default=DEFAULT_BASE_IMAGE_PATH,
                        help='Base path to search for folders with images.')
    parser.add_argument('-n', '--number_of_images', type=int, default=DEFAULT_NUM_IMAGES,
                        help='Number of images to use in the composite.')
    parser.add_argument('--shuffle', type=bool, default=True, help='Shuffle images before processing.')

    args = parser.parse_args()

    hostname = socket.gethostname()

    try:
        image_folder = find_latest_image_folder(args.base_path)
    except Exception as e:
        print(f"Error: {e}")
        logger.error(f"Failed to find image folder: {e}")
        sys.exit(1)

    total_frames = generate_nest_image(image_folder, args.number_of_images, hostname, shuffle=args.shuffle)

    duration = round(time.time() - start_time, 2)
    print(f"Finished. Processed {total_frames} images in {duration} seconds.")
    logger.info(f"Generated nest image in {duration}s using {total_frames} images.")

if __name__ == "__main__":
    main()
    logging.shutdown()
