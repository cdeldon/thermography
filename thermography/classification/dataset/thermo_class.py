from typing import List


class ThermoClass:
    def __init__(self, class_name: str, class_value: int, class_folder: str = None):
        self.class_name = class_name
        self.class_value = class_value

        if class_folder is None:
            class_folder = self.class_name
        self.class_folder = class_folder


ThermoClassList = List[ThermoClass]
