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

def create_folder_if_needed(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path

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

    imgdata = cv2.imread(image_paths[0])
gray_img = cv2.cvtColor(imgdata, cv2.COLOR_RGB2GRAY)
h, w = gray_img.shape
x = da.zeros((total_frames, h, w), dtype='uint8')

for idx, img_path in enumerate(image_paths):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Could not load image {img_path}, skipping.")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if gray.shape != (h, w):
        gray = cv2.resize(gray, (w, h))
    x[idx] = gray

median_img = da.median(x, axis=0).compute()

create_folder_if_needed(output_folder)

    filename = f"{hostname}-{date_str}-nest_image.png"
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, median_img)

    print(f"Saved composite image to: {output_path}")
    logger.info(f"Composite image created at {output_path} using {total_frames} images.")
    return total_frames

def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(description="Generate nest images per date per mc folder.")
    parser.add_argument('-b', '--base_path', type=str, default=DEFAULT_BASE_IMAGE_PATH,
                        help='Base path to search for mc/date folders with images.')
    parser.add_argument('-n', '--number_of_images', type=int, default=DEFAULT_NUM_IMAGES,
                        help='Number of images to use per composite image.')
    parser.add_argument('--shuffle', type=bool, default=True, help='Shuffle images before processing.')

    args = parser.parse_args()
    base_path = args.base_path
    num_images = args.number_of_images
    shuffle = args.shuffle

    hostname = socket.gethostname()

    mc_folders = glob.glob(os.path.join(base_path, '*'))  # all mc folders under round1

    if not mc_folders:
        print(f"No mc folders found in {base_path}")
        sys.exit(1)

    total_images_processed = 0
    total_composites = 0

    for mc_folder in mc_folders:
        if not os.path.isdir(mc_folder):
            continue

        print(f"\nProcessing mc folder: {mc_folder}")

        date_folders = glob.glob(os.path.join(mc_folder, '*'))  # all date folders inside mc_folder
        if not date_folders:
            print(f"No date folders found inside {mc_folder}")
            continue

        nest_images_root = os.path.join(mc_folder, "Nest Images")
        create_folder_if_needed(nest_images_root)

        for date_folder in date_folders:
            if not os.path.isdir(date_folder):
                continue

            date_str = os.path.basename(date_folder)
            output_folder = os.path.join(nest_images_root, date_str)
            create_folder_if_needed(output_folder)

            images_used = generate_nest_image(date_folder, num_images, hostname, date_str, output_folder, shuffle)
            if images_used > 0:
                total_images_processed += images_used
                total_composites += 1

    elapsed = round(time.time() - start_time, 2)
    print(f"\nProcessed {total_composites} composite images using a total of {total_images_processed} images in {elapsed} seconds.")
    logger.info(f"Processed {total_composites} composite images using {total_images_processed} images in {elapsed}s.")

if __name__ == "__main__":
    main()
    logging.shutdown()
