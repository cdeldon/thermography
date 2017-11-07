import os


def create_directory_list(root_dir: str):
    """
    Creates a list of directories for dataset loading.
    :param root_dir: Absolute path to the root directory of the dataset.

    The dataset root directory must be of the following form:
    ::
       root_dir
       |__video1
       |    |__0-1000
       |    |__1000_2000
       |__video2
       |    |__0-500
       |    |__500-1000
       |    |__1000-1200
       |__video3
            |__0-1000

    and each folder 'xxxx-yyyy' must contain three directories associated to the classes of the dataset.

    :return: A list of absolute paths to the class directories containing the dataset images.
    """
    if not os.path.exists(root_dir):
        raise FileNotFoundError("Directory {} does not exist".format(root_dir))

    # List all directories associated to different videos.
    recording_path_list = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]

    input_data_path = []
    for g in recording_path_list:
        # Append the different directories associated to different video frame intervalls.
        input_data_path.extend([os.path.join(g, f) for f in os.listdir(g)])

    return input_data_path
