import cv2
import os


def warp_img():
    for i, f in enumerate(os.listdir("bkg")):
        img = cv2.imread("bkg/" + f)
        im = cv2.resize(img, (28, 28))
        cv2.imwrite("data/" + str(i+1) + ".jpg", im)


if __name__ == "__main__":
    warp_img()

