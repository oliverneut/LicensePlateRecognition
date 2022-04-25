import cv2
import numpy as np
import pandas as pd
import Localization
import Recognize

"""
In this file, you will define your own CaptureFrame_Process funtion. In this function,
you need three arguments: file_path(str type, the video file), sample_frequency(second), save_path(final results saving path).
To do:
1. Capture the frames for the whole video by your sample_frequency, record the frame number and timestamp(seconds).
2. Localize and recognize the plates in the frame.(Hints: need to use 'Localization.plate_detection' and 'Recognize.segmetn_and_recognize' functions)
3. If recognizing any plates, save them into a .csv file.(Hints: may need to use 'pandas' package)
Inputs:(three)
1. file_path: video path
2. sample_frequency: second
3. save_path: final .csv file path
Output: None
"""

FULL_PATH = '/home/imageprocessingcourse'

# samples the video with the given sample frequency, adds the selected frames to a list together with the index
def sample_video(file_path, sample_frequency):
    video = cv2.VideoCapture(file_path)

    sample_frequency = 2
    # get the amount of frames per second of the video
    fps = video.get(cv2.CAP_PROP_FPS)
    video_arr = []

    sum = 1
    captured_frames_count = 0
    while video.isOpened():
        ret, frame = video.read()

        if ret:
            if sum % (fps/sample_frequency) == 0:
                captured_frames_count += 1
                seconds = sum / fps
                video_arr.append((sum, frame, seconds))
        else:
            break
        sum += 1

    video.release()

    print("FPS of video :", fps)
    print("Number of frames in video :", sum)
    print("Number of frames captured :", captured_frames_count)

    return video_arr


def valid(plate):
    text = plate[1]

    for i in range(0, len(text) - 1):
        if text[i] == '-' or text[i + 1] == '-':
            continue
        if text[i].isalpha() == text[i + 1].isalpha():
            continue
        if text[i].isalpha() != text[i + 1].isalpha():
            return False

    return True


def filter_plates(plates):
    final_plates = []
    for lp in plates:
        license_plate_number = ""
        if lp[2]:
            if valid(lp):
                license_plate_number = lp[1]
            else:
                license_plate_number = ""
        else:
            license_plate_number = ""

        output = np.array([license_plate_number, lp[0], lp[3]])
        final_plates.append(output)

    return final_plates


def CaptureFrame_Process(file_path, sample_frequency, save_path):
    # the video is sampled with a sample_frequency specified in the arguments of main.py
    video_arr = sample_video(file_path, sample_frequency)

    localized_plates = Localization.plate_detection(video_arr)
    recognized_plates = Recognize.segment_and_recognize(localized_plates)

    final_plates = filter_plates(recognized_plates)

    df = pd.DataFrame(final_plates, columns=['License plate', 'Frame no.', 'Timestamp(seconds)'])
    df.to_csv(save_path, index=None)  # 'record.csv'



    