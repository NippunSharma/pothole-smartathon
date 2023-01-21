import numpy as np

from .cost_mendonca_cipolla import cost_mendonca_cipolla
from .cost_kruppa_classical import cost_kruppa_classical

from scipy.io import savemat
import logging
from scipy.optimize import least_squares
import cv2

def calibrate(F, A):
    '''
    Parameters:
        F -> Fundamental matrices for each image pair.
        A -> Approximation to intrinsic matrix.
    Returns:
        K -> Refined intrinsic matrix.
    '''
    logging.debug("Starting calibration.")

    X0 = np.array([A[0,0], A[0,1], A[0,2], A[1,1], A[1,2]])

    K_MC1 = least_squares(lambda X: cost_mendonca_cipolla(F, X, "1"),
                          X0, method="lm", xtol=1e-10)

    logging.debug(f"Intrinsics computed by Mendonca and Cipolla (Method 1): {K_MC1.x}")

    K_MC2 = least_squares(lambda X: cost_mendonca_cipolla(F, X, "2"),
                          X0, method="lm", xtol=1e-10)

    logging.debug(f"Intrinsics computed by Mendonca and Cipolla (Method 2): {K_MC2.x}")

    K_CK = least_squares(lambda X: cost_kruppa_classical(F, X),
                         X0, method="lm", ftol=1e-10, xtol=1e-10)
    
    logging.debug(f"Intrinsics computed by Classical Kruppa: {K_CK.x}")

    print({
        "mc-1": K_MC1.x,
        "mc-2": K_MC2.x,
        "ck": K_CK.x
    })

    return {
        "mc-1": K_MC1.x,
        "mc-2": K_MC2.x,
        "ck": K_CK.x
    }

def get_skew_mat(vec):
    vec = vec.ravel()
    vec = np.concatenate([vec, np.array([1])])
    return np.array([[0, -vec[2], vec[1]], [vec[2], 0, -vec[0]], [-vec[1], vec[0], 0]])

def calibrate_video(video_path, K_approx, start_second, num_frames):
    '''
    Perform calibration through a video.

    Parameters:
        video_path -> path to video file.
        K_approx -> initial approximation to intrinsic matrix.
        start_second -> start time (in seconds) to shortlist the frames.
        num_frames -> number of (almost) consecutive frames to extract from the image.
    Returns:
        intrinsic matrices calculated by various algorithms.
    '''

    # load video.
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if not cap.isOpened():
        logging.error(f"Could not open video file {video_path}")
        return

    logging.debug(f"read file {video_path} successfully.")

    shortlisted_frames = []
    frame_count = 0

    while(cap.isOpened() and len(shortlisted_frames) < num_frames):
        ret, frame = cap.read()

        if ret:
            frame_count += 1
            if start_second > fps * frame_count:
                continue

            if frame_count % 5 == 0:
                shortlisted_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        else:
            break

    cap.release()

    # for each pair of shortlisted frames, extract features and estimate
    # the fundamental matrix.
    Fs = np.zeros(shape=(3,3,num_frames,num_frames))

    for i in range(num_frames):
        for j in range(num_frames):
            if i == j:
                continue

            img1 = shortlisted_frames[i]
            img2 = shortlisted_frames[j]

            orb = cv2.ORB_create(3000)
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

            kp1, des1 = orb.detectAndCompute(img1, None)
            kp2, des2 = orb.detectAndCompute(img2, None)
            matches = flann.knnMatch(des1,des2,k=2)

            # Find the matches there do not have a to high distance
            good = []
            try:
                for m, n in matches:
                    if m.distance < 0.8 * n.distance:
                        good.append(m)
            except ValueError:
                pass

            # Get the image points form the good matches
            q1 = np.float32([kp1[m.queryIdx].pt for m in good])
            q2 = np.float32([kp2[m.trainIdx].pt for m in good])

            F, _ = cv2.findFundamentalMat(q1,q2,cv2.FM_LMEDS)
            Fs[:, :, i, j] = F

    savemat("my_data.mat", {
        "Fs": Fs,
        "A": K_approx
    })

    return calibrate(Fs, K_approx)
