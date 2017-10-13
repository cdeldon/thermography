import thermography as tg
import os


def _main():
    SETTINGS_DIR = tg.settings.get_settings_dir()
    camera_param_file = os.path.join(SETTINGS_DIR, "camera_parameters.json")

    tg.settings.set_data_dir("Z:/SE/SEI/Servizi Civili/Del Don Carlo/termografia/")
    IN_FILE_NAME = os.path.join(tg.settings.get_data_dir(), "Ispez Termografica Ghidoni 1.mov")

    app = tg.App(input_video_path=IN_FILE_NAME, camera_param_file=camera_param_file)

    app.load_video(start_frame=1500, end_frame=1800)
    app.run()


if __name__ == '__main__':
    _main()
