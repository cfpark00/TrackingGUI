import argparse
from src.gui import GUI
from src.tools.simple_helpers import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launch Annotation GUI')
    parser.add_argument('file_path', help='file path')
    parser.add_argument('--settings_file_path',default="settings.txt", help='settings file path')
    parser.add_argument('--mode',default="points", help='settings file path')

    args=parser.parse_args()

    assert args.mode in ["points","onclick"]

    file_path=args.file_path
    settings_file_path=args.settings_file_path

    settings=load_settings(settings_file_path)
    settings["mode"]=args.mode
    gui=GUI(file_path,settings)
    gui.start()
    print("GUI closed succesfully")
