import os
import datetime

def get_free_filename(date_str = None):
    """
    Returns a filename for a file that should be followed by the current date,
    but if this is already in the frames folder, then iterate by 1, until
    you have a free file name.
    """
    base_filename = "d"
    frames_folder = "data/new"
    date_str = datetime.date.today().strftime("%Y-%m-%d") if date_str is None else date_str
    base_filename_with_date = f"{base_filename}_{date_str}"

    i = 0
    while True:
        filename = os.path.join(frames_folder, f"{base_filename_with_date}_{i}")
        if not os.path.exists(filename):
            return filename
        i += 1