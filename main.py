import cv2 as cv
import numpy as np
import img_get
import gc


def cvtimg(img, zoom=1.0):
    if zoom != 1:
        w = int(img.width * zoom)
        h = int(img.height * zoom)
        img = img.resize((w, h))
    img = img.convert("RGB")
    img = cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)
    return img


delay_frams_per_scene = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 每个场景延时

if __name__ == "__main__":

    start_scene = int(input("开始场景: "))

    while True:
        try:
            for scene in range(len(img_get.r_fms)):
                gc.collect()
                if scene < start_scene:
                    continue

                show_imgs = img_get.get_frames(scene)
                show_delays = img_get.get_frames_delay(scene)

                count = 0
                for frames in show_imgs:
                    cv.imshow("kaibai", cvtimg(frames, 0.5))

                    if count > len(show_delays) - 1:
                        print(f"scene: {scene}, frame: {count} - ot")
                        _wtk = 50
                    else:
                        _wtk = int(show_delays[count] * 1000)
                    cv.waitKey(_wtk)
                    count += 1

                if scene > len(delay_frams_per_scene) - 1:
                    print(f"scene: {scene} - ot")
                    _wtk = 50
                else:
                    _wtk = img_get.calc_fdelay_ms(delay_frams_per_scene[scene])

            count = 0
            while True:
                im = cv.imread("./src/kaibai.png", cv.IMREAD_COLOR)
                cv.imshow("kaibai", im)
                cv.waitKey(30)
                count += 1
                if count > 100:
                    break

            cap = cv.VideoCapture("./src/截取.mp4")
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    cv.imshow("kaibai", frame)
                else:
                    break
                key = cv.waitKey(25)

            break

        except img_get.BaiLanError as sb:
            print(sb)
