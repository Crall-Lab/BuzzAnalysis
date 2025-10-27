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

# ---- CONFIG ----
DEFAULT_DATA_FOLDER_PATH = os.path.expanduser("~/bumblebox_data")
DEFAULT_NUMBER_OF_IMAGES = 50
LOG_PATH = os.path.expanduser("~/bumblebox_logs/log.log")

def is_date_format(folder_name):
    try:
        datetime.datetime.strptime(folder_name, "%Y-%m-%d")
        return True
    except ValueError:
        return False

# ---- Logging ----
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logging.basicConfig(filename=LOG_PATH, encoding='utf-8',
                    format='%(filename)s %(asctime)s: %(message)s',
                    filemode='a', level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# ---- Functions ----
def create_todays_folder(base_path):
    today = datetime.date.today().isoformat()
    folder_path = os.path.join(base_path, today)
    try:
        os.makedirs(folder_path, exist_ok=True)
        return 0, folder_path
    except Exception as e:
        logger.error(f"Error creating folder: {e}")
        return 1, folder_path

def make_nest_images_dir(base_path):
    nest_dir = os.path.join(base_path, "Nest Images")
    try:
        os.makedirs(nest_dir, exist_ok=True)
        return 0, nest_dir
    except Exception as e:
        print(f"Failed to make nest image directory: {e}")
        return 1, nest_dir

def generate_nest_image(todays_folder_path, today, number_of_images, hostname, shuffle=True):
    ret, nestpath = make_nest_images_dir(os.path.dirname(todays_folder_path))
    files = glob.glob(f"{todays_folder_path}/*.png")
    if shuffle:
        random.shuffle(files)
    
    files = files[:number_of_images]
    if not files:
        print("No PNG files found.")
        return 0

    print(f"Using {len(files)} images for composite.")

    imgdata = cv2.imread(files[0])
    gray_img = cv2.cvtColor(imgdata, cv2.COLOR_BGR2GRAY)

    h, w = gray_img.shape
    x = da.zeros((len(files), h, w), dtype='uint8')

    for index, file in enumerate(files):
        try:
            imgdata = cv2.imread(file)
            gray_img = cv2.cvtColor(imgdata, cv2.COLOR_BGR2GRAY)
            gray_img = cv2.resize(gray_img, (w, h))  # Ensure uniform size
            x[index] = gray_img
        except Exception as e:
            print(f"Error reading {file}: {e}")

    composite = da.median(x, axis=0).compute()
    out_path_1 = os.path.join(todays_folder_path, f"{hostname}-{today}_nest_image.png")
    out_path_2 = os.path.join(nestpath, f"{hostname}-{today}_nest_image.png")

    cv2.imwrite(out_path_1, composite)
    cv2.imwrite(out_path_2, composite)

    print(f"Composite image saved to:\n{out_path_1}\n{out_path_2}")
    logger.debug(f"Composite image written using {len(files)} images.")
    return len(files)

# ---- Main ----
def main():
    start = time.time()

    parser = argparse.ArgumentParser(
        prog='Generate composite nest images for each date folder in a parent directory'
    )
    parser.add_argument(
        '-p', '--data_folder_path',
        type=str,
        required=True,
        help='Path to the parent folder that contains dated subfolders (e.g., ~/bumblebox_data)'
    )
    parser.add_argument(
        '-i', '--number_of_images',
        type=int,
        default=50,
        help='Number of images to use per date folder'
    )
    parser.add_argument(
        '-sh', '--shuffle',
        type=bool,
        default=True,
        help='Shuffle images before selection'
    )
    args = parser.parse_args()

    parent_folder = os.path.expanduser(args.data_folder_path)
    hostname = socket.gethostname()

    if not os.path.exists(parent_folder):
        print(f"Parent folder does not exist: {parent_folder}")
        return

    # List subfolders that match a date pattern (YYYY-MM-DD)
    date_folders = sorted([
        f for f in os.listdir(parent_folder)
        if os.path.isdir(os.path.join(parent_folder, f)) and is_date_format(f)
    ])

    print(f"Found {len(date_folders)} date folders in: {parent_folder}")
    if not date_folders:
        print("No valid date folders found. Exiting.")
        return

    for date_folder in date_folders:
        full_path = os.path.join(parent_folder, date_folder)
        print(f"\n🔧 Processing: {full_path}")

        pngs = glob.glob(os.path.join(full_path, "*.png"))
        if not pngs:
            print(f"⚠️ No PNG files in {full_path}, skipping.")
            continue

        first_filename = os.path.basename(pngs[0])
        try:
            extracted_hostname = first_filename.split("_")[0]
        except:
            print(f"⚠️ Could not extract hostname from filename: {first_filename}, skipping.")
            continue

        try:
            total_frames = generate_nest_image(
                todays_folder_path=full_path,
                today=date_folder,
                number_of_images=args.number_of_images,
                hostname=extracted_hostname,
                shuffle=args.shuffle
            )
            print(f"✅ Done: {total_frames} frames used for {date_folder}")
        except Exception as e:
            print(f"❌ Error processing {date_folder}: {e}")

    end = time.time()
    print(f"\n🎉 All done! Total time: {round(end - start, 2)} seconds.")



if __name__ == '__main__':
    main()
    logging.shutdown()
