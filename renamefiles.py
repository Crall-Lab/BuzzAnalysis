import os
import re


# Ask the user for the folder path
root = input("Enter the path to the folder: ").strip()

# Check if the path exists
if not os.path.isdir(root):
    print("The path you entered does not exist or is not a directory.")
else:
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.startswith("bumblebox-") and f.endswith(".csv") or f.endswith(".mp4") or f.endswith(".png"):
            # Match date _ HH_MM_SS pattern, only replace inside the time part
                new_name = re.sub(
                    r'_(\d{4}-\d{2}-\d{2})_(\d{2})_(\d{2})_(\d{2})',
                    lambda m: f"_{m.group(1)}_{m.group(2)}-{m.group(3)}-{m.group(4)}",
                    f
                )
                if new_name != f:
                    old_path = os.path.join(dirpath, f)
                    new_path = os.path.join(dirpath, new_name)
                    os.rename(old_path, new_path)
                    print(f"Renamed: {f} -> {new_name}")
