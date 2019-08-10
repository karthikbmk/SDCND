from csv import DictReader
import cv2
import json

class Helper:
    def __init__(self):
        pass

    def csv_to_list(self, path):
        reader = None
        lines = []

        with open(file=path, mode='r') as f:
            reader = DictReader(f)
            for line in reader:
                lines.append(dict(line))

        return lines

    def show_image(self, img):
        cv2.imshow('test', img)
        cv2.waitKey()


    def load_json(self,js_path):

        with open(js_path, 'r') as f:
            js = json.load(f)
            return js