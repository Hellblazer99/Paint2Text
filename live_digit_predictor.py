import win32gui
from subprocess import Popen
from mss import mss, tools
import cv2 as cv
import numpy as np
from PIL import Image
from make_preds import PredictDigit


class WindowImage(PredictDigit):

    def __init__(self):
        super().__init__()
        self.window_title = "Untitled - Paint"
        self.sct = mss()

    def open_paint(self):
        Popen("mspaint")

    def close_paint(self):
        Popen("taskkill /f /IM mspaint.exe")

    def get_coord(self):
        hwnd = self.get_window_handle()
        resize = win32gui.MoveWindow(hwnd, 0, 0, 430, 520, False)
        rect = win32gui.GetWindowRect(hwnd)
        x = rect[0]
        y = rect[1]
        w = rect[2] - x
        h = rect[3] - y
        return (x+10), (y+60), (h-120), (w-30)

    def get_window_handle(self):
        while not win32gui.FindWindow(None, self.window_title):
            pass
        return win32gui.FindWindow(None, self.window_title)

    def stream_screen(self):
        """
        This method has to fill self._monitors with all information
        and use it as a cache:
            self._monitors[0] is a dict of all monitors together
            self._monitors[N] is a dict of the monitor N (with N > 0)
        """
        self.open_paint()

        cv.namedWindow("Stream")

        while 1:
            x, y, h, w = self.get_coord()
            # The screen part to capture
            monitor = {"top": y, "left": x, "width": w, "height": h}
            scr_shot = self.sct.grab(monitor)
            img = np.array(Image.frombytes('RGB', (scr_shot.size[0], scr_shot.size[1]), scr_shot.rgb))
            # img = cv.imdecode(np.frombuffer(tools.to_png(scr_shot.rgb, scr_shot.size), dtype=np.uint8), cv.IMREAD_COLOR)

            img = cv.bitwise_not(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
            # print(img.shape)
            cv.imshow("Stream", img)
            self.run_preds(cv.resize(img, (28, 28)))
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        cv.destroyAllWindows()
        self.close_paint()

    def capture_paint(self):
        with self.sct:
            x, y, h, w = self.get_coord()
            # The screen part to capture
            monitor = {"top": y, "left": x, "width": w, "height": h}
            output = "sct-{top}x{left}_{width}x{height}.png".format(**monitor)

            # Grab the data
            sct_img = self.sct.grab(monitor)

            # Save to the picture file
            tools.to_png(sct_img.rgb, sct_img.size, output=output)
            print(output)


if __name__ == '__main__':
    wind_obj = WindowImage()
    wind_obj.stream_screen()
