class ObjectDetection:
    def __init__(self, cls: str, conf: float, x: float, y: float, w: float, h: float):
        self.cls = cls
        self.conf = conf
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def __str__(self):
        return f"cls: {self.cls}, conf: {self.conf}, x: {self.x}, y: {self.y}, w: {self.w}, h: {self.h}"
    def __repr__(self):
        return self.__str__()
