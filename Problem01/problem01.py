import os
import cv2
import numpy as np


def histEq(image):

    # converting the image to YUV format
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    # separating the Y channel (Brightness channel)
    img_y = img_yuv[:, :, 0]

    flat_ch = img_y.flatten()

    bins = range(257)

    # categoring the intensites into 0-255
    values, bins = np.histogram(flat_ch, bins)

    # Dividing by the number of pixels
    normalized_inten = values/(len(flat_ch))

    cum_sum = np.cumsum(normalized_inten)

    # calculating the cdf for the values
    cdf = np.floor(cum_sum*255)

    # constructing the cdf image by matching the intensity from original image
    final_img = np.array([cdf[pixel] for pixel in flat_ch], dtype=np.uint8)

    img_y = final_img.reshape(img_y.shape)

    img_yuv[:, :, 0] = img_y

    # convering back to BGR format
    img_final = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    return img_final


def adaptiveHist(img):

    for i in range(0, img.shape[0], int(img.shape[0]/8)):
        for j in range(0, img.shape[1], int(img.shape[1]/8)):
            img[i:i+int(img.shape[0]/8), j:j+int(img.shape[1]/8)
                ] = histEq(img[i:i+int(img.shape[0]/8), j:j+int(img.shape[1]/8)])

    return img


if __name__ == '__main__':

    path = '.'
    dir_list = os.listdir(path)

    for item in dir_list:
        if not(item.endswith(".png")):
            dir_list.remove(item)
    flag = 1
    dir_list.sort()

    print("press q to end the videos.")

    fourcc = cv2.VideoWriter_fourcc('m','p','4', 'v')
    out1 = cv2.VideoWriter("Histogram_Eq.mp4", fourcc, 5, (1224, 370))
    out2 = cv2.VideoWriter("Adaptive_hist_eq.mp4", fourcc, 5, (1224, 370))

    # while flag:

    for img in dir_list:
        image = cv2.imread(img)

        cv2.imshow("Input", image)

        Hist_EQ = histEq(image)

        Adap_hist = adaptiveHist(image)

        cv2.imshow("Histogram Equalization", Hist_EQ)

        cv2.imshow("Adaptive Histogram Equalization", Adap_hist)

        out1.write(Hist_EQ)
        out2.write(Adap_hist)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            flag = 0
            break

    out1.release()
    out2.release()
    cv2.destroyAllWindows()
