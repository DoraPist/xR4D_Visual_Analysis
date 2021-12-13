from pymongo import MongoClient
import time
import requests
from bson.json_util import ObjectId
import json
from web_functions import WebFunctions
import cv2
import os
import predictors
from collections import Counter
from keras.models import load_model
from datetime import datetime
import tensorflow as tf
import numpy as np
import subprocess
# from pixellib.instance import instance_segmentation
from pixellib.semantic import semantic_segmentation
import cv2
from imutils import paths
import imutils
from os.path import exists
import matplotlib.pyplot as plt
import logging
from PST.model import WCT2
from PST.data_processing import build_input_pipe, restore_image
from PST.utils import http_get_img, get_local_img, display_outputs

# ============= For log file ============= #
lf = open("logfilename.log", "w")   # Create a new log file to keep record only for current analysis

# Define log file - it will contain anything printed in console
logging.basicConfig(filename="logfilename.log", level=logging.INFO)

MONGO_URI = "mongodb+srv://dora_user:wk9NmSQIPqgXNf8O@doracluster.se2tb.mongodb.net/ImagesDB?ssl=true&ssl_cert_reqs=CERT_NONE"  # Atlas MongoDB
# MONGO_URI = "mongodb://xr4d_visuals_user:B6^M2qAe@xr4drama.iti.gr:27017/XR4D_Visual_Analysis_DB"  # Local MongoDB
db = MongoClient(MONGO_URI)


def detect_blur_fft(image, size=60, thresh=10):
    # grab the dimensions of the image and use the dimensions to
    # derive the center (x, y)-coordinates
    (h, w) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))

    # compute the FFT to find the frequency transform, then shift
    # the zero frequency component (i.e., DC component located at
    # the top-left corner) to the center where it will be more
    # easy to analyze

    fft = np.fft.fft2(image)
    fftShift = np.fft.fftshift(fft)

    # zero-out the center of the FFT shift (i.e., remove low
    # frequencies), apply the inverse shift such that the DC
    # component once again becomes the top-left, and then apply
    # the inverse FFT
    fftShift[cY - size:cY + size, cX - size:cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)

    # compute the magnitude spectrum of the reconstructed image,
    # then compute the mean of the magnitude values
    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)
    # the image will be considered "blurry" if the mean value of the
    # magnitudes is less than the threshold value
    return (mean, mean <= thresh)


def visual_analysis(simmoid, sendtoKB, sendtoRec, pst):

    # Update request's status in MongoDB
    # db.ImagesDB.InputQueue.update_one(filter={"input.simmoid": input["simmoid"]}, update= {"$set": {"status": 1}})  # 1: In Progress
    # print("Input with simmoid {} is being analyzed!")

    # visual_analysis(input)
    print("[INFO] Analyzing input with simmoid {}. Please wait...".format(simmoid))
    logging.info("Analyzing input with simmoid {}. Please wait...".format(simmoid))

    # ------------------------------------------------------------------------------ #
    # ----------------------------- Vusual Analysis -------------------------------- #
    # ------------------------------------------------------------------------------ #
    # time.sleep(2)  # Simulating visual analysis
    # test()  # Testing the usage of functions from other py files

    # Get request from MongoDB by simmoid
    input_request = db.ImagesDB.InputQueue.find_one(filter={"input.simmoid": simmoid})['input']
    # input_request = db.XR4D_Visual_Analysis_DB.InputQueue.find_one(filter={"input.simmoid": simmoid})['input']
    # print(input_request)

    # Check entity: image, video or twitter_post?
    entity = input_request['entity']
    print("[INFO] Entity: ", entity)
    logging.info("Entity: ", entity)

    # Get and keep project_id to send to KB, (3D rec and geoserver?)
    project_id = input_request['project_id']
    print("project id: ", project_id)
    logging.info("project id: ", project_id)

    ####################################################################################################################
    # ======================  TWITTER POST  ======================== #                                      TWITTER POST
    ####################################################################################################################
    # Check if the input is a twitter post and proceed with proper analysis
    flag_twitter = False
    entities = []
    urls = []
    if entity == "twitter_post":
        print("[INFO] Analyzing input twitter post...")
        logging.info("Analyzing input twitter post...")

        flag_twitter = True

        # simmo_id = request_json["simmoid"]

        # Get JSON file with info
        r = WebFunctions.get_json(simmoid, entity)
        input_json = r.json()
        # print(input_json)
        items = input_json["items"]
        len_items = len(items)
        print("Length of items list", len(items))
        logging.info("Length of items list", len(items))
        for i in range(1, len_items):
            current_item = items[i][0]
            dictionary = current_item[1]
            print(dictionary["url"])
            logging.info(dictionary["url"])
            print(dictionary["type"])
            logging.info(dictionary["type"])
            urls.append(dictionary["url"])
            entities.append(dictionary["type"])

    ####################################################################################################################
    # ====================== IMAGE ANALYSIS ======================== #                                             IMAGE
    ####################################################################################################################
    # Check if the input is an image, a video or a twitter post and proceed with proper analysis
    if entity == "image" or "image" in entities or "IMAGE" in entities:
        print("[INFO] Analyzing input image...")
        logging.info("Analyzing input image...")

        # Check if entity comes from a Twitter Post:
        if flag_twitter:

            # Defile subpaths
            image_path = "Data/" + entity + "_" + simmoid + "/"
            only_filename = simmoid + ".jpg"
            save_filename = image_path + only_filename  # This is the path/filename where the downloaded video will be saved  # ToDo: Folders

            # Create directory to save image if it does not exist
            if not os.path.exists(image_path):
                os.makedirs(image_path)
            # Download image (# TODO: This works only for 1 image/item now - Do for multiple!!!
            url = urls[0]

            print("[INFO] Image will be downloaded from ", url)

            # Count time to download video and split it in frames
            t1 = time.time()

            # --------------------> Download image
            WebFunctions.download_from_url(url, image_path, only_filename)  # TODO: uncommend this once server is up

            # # TODO: commend this ---------------
            # online_image = requests.get(url)
            # print(online_image)
            #
            # if online_image.status_code == 200:
            #     open(save_filename, 'wb').write(online_image.content)
            # # TODO: ------------------------------

            image = cv2.imread(save_filename)
            print("[INFO] Loaded image: ", simmoid + ".jpg")
            logging.info("Loaded image: ", simmoid + ".jpg")

        # --------------------- Create JSON for KB  -----------------------#
        # For KB - Information only per video shot
        KB_JSON = {}
        KB_JSON["header"] = {"timestamp": str(datetime.now(tz=None)), "sender": "Visual Analysis", "entity": entity,
                             "simmoid": simmoid, "project_id": project_id}  # ToDo
        KB_JSON["shotInfo"] = []

        # Save extracted info for each image for KB
        shot_info_dict_KB = {}
        shot_info_dict_KB["shotIdx"] = 0
        shot_info_dict_KB["startFrame"] = 0
        shot_info_dict_KB["endFrame"] = 0
        # shot_info_dict_KB["area"] = []
        # shot_info_dict_KB["areaProb"] = []
        # shot_info_dict_KB["outdoor"] = []
        # shot_info_dict_KB["emergencyType"] = []
        # shot_info_dict_KB["emergencyProb"] = []
        shot_info_dict_KB["objectsFound"] = []
        shot_info_dict_KB["peopleInDanger"] = 0
        shot_info_dict_KB["vehiclesInDanger"] = 0
        shot_info_dict_KB["riverOvertop"] = False

        # --------------------- Load ML models -----------------------#

        # load model for SR   # ToDo: way not load model everytime?
        SR_model = load_model('Trained_models/SR/xr4drama_places_model_ep5_bs32.h5')
        print("[INFO] SR model is successfully loaded!")
        logging.info("SR model is successfully loaded!")

        # load model for EmC
        EmC_model = load_model('Trained_models/EmC/vgg16_places2_lastconv_19ep_1e-03lr_MME2017.pkl')
        print("[INFO] EmC model is successfully loaded!")
        logging.info("EmC model is successfully loaded!")

        if pst:
            # load model for PST
            # train_tfrec = "PST/tfrecords/train.tfrec"
            # val_tfrec = "PST/tfrecords/val.tfrec"

            model = WCT2(image_size=None, lr=1e-4, gram_loss_weight=1.0)
            model.wct.load_weights(model.checkpoint_path)
            # model.train(train_tfrec, epochs=10, batch_size=8)

            model.wct.save_weights(model.checkpoint_path)

            print("[INFO] PST model is successfully loaded!")
            logging.info("PST model is successfully loaded!")

        # load model for BOL
        semantic_segmentation_model = semantic_segmentation()
        semantic_segmentation_model.load_ade20k_model("Trained_models/BOL/Pixellib/deeplabv3_xception65_ade20k.h5")
        print("[INFO] BOL model is successfully loaded!")
        logging.info("BOL model is successfully loaded!")

        # --------------------- Scene Recognition (SR)-----------------------#

        # Run SR model
        scene, sr_prob, is_outdoor = predictors.SR_model(image, SR_model)
        # SR_model.clear

        shot_info_dict_KB["area"] = scene
        shot_info_dict_KB["areaProb"] = sr_prob/100
        shot_info_dict_KB["outdoor"] = is_outdoor

        # --------------------- Emergency Classification (EmC) ---------------------#

        # Run EmC model
        emergency, emc_prob = predictors.EmC_model(image, EmC_model)

        shot_info_dict_KB["emergencyType"] = emergency
        shot_info_dict_KB["emergencyProb"] = np.float64(emc_prob)
        # emergencies.append(emergency)
        # emc_probs.append(emc_prob)

        # Check if we will use PST or not
        if pst:
            # --------------------- Photorealistic Style Transfer (PST) ---------------------#
            # rst = None
            rst = 64 * 14  # (896)
            # test_id = np.random.randint(1, 60)
            # test_id = 20

            # start = datetime.datetime.now()
            # content = get_local_img("./examples/input/tar55.png".format(test_id), rst)
            content = get_local_img(image_path + str(simmoid) + ".jpg", rst)
            style = get_local_img("PST/examples/style/tar44.png", rst)

            output = model.transfer(tf.cast(content, tf.float32), tf.cast(style, tf.float32), 0.8)

            plt.imsave(image_path + str(simmoid) + "_pst.jpg", output[0] / 255.0)

            # -------------- Building and Object Localization (BOL) ---------------#
            # Run BOL model
            segvalues, objects_masks, image_overlay = semantic_segmentation_model.segmentAsAde20k(
                image_path + str(simmoid) + "_pst.jpg", overlay=False, extract_segmented_objects=True,
                output_image_name=image_path + str(simmoid) + "_mask.png")
        else:

            # -------------- Building and Object Localization (BOL) ---------------#

            # Run BOL model
            # segvalues, segoverlay = semantic_segmentation_model.segmentAsAde20k(
            #     image_path + str(simmoid) + ".jpg", overlay=False,
            #     output_image_name=image_path + str(simmoid) + "_mask.png")
            segvalues,image_overlay = semantic_segmentation_model.segmentAsAde20k(
                image_path + str(simmoid) + ".jpg", overlay=False,
                output_image_name=image_path + str(simmoid) + "_mask.png")

        for l in range(0, len(segvalues['class_names'])):
            objects_dict = {'type': segvalues['class_names'][l], 'probability': segvalues['ratios'][l] / 100}
            shot_info_dict_KB['objectsFound'].append(objects_dict)

        # print(frames_path + "frame_" + str(frame_idx) + ".jpg")

        KB_JSON["shotInfo"].append(shot_info_dict_KB)

        print(KB_JSON["shotInfo"])
        logging.info(KB_JSON["shotInfo"])

        # Save KB_JSON file
        with open(image_path + 'KB.json', 'w') as f:
            json.dump(KB_JSON, f)

            # Save output for KB (maybe this will be inside the analysis function)
        db.ImagesDB.VisualsOutputKB.insert_one(KB_JSON)
        # db.XR4D_Visual_Analysis_DB.VisualsOutputKB.insert_one(KB_JSON)

    #################################################################################################################
    # ====================== VIDEO ANALYSIS ======================== #                                         VIDEO
    #################################################################################################################
    # Check if the input is a video and proceed with proper analysis
    if entity == "video" or "video" in entities or "VIDEO" in entities:
        print("[INFO] The analysis for input video is starting...")
        logging.info("The analysis for input video is starting...")

        if flag_twitter == True:
            url = urls[0]  # ToDo: This supports only 1 video from Twitter <--- support > 1
        else:
            # TODO ---> uncomment once xr4drama server is up
            # simmo_id = request_json["simmoid"]

            # Get JSON file with info
            r = WebFunctions.get_json(simmoid, entity)
            input_json = r.json()
            print(input_json)
            logging.info(input_json)

            # Get video url
            if "alternativeUrl" in input_json.keys():
                if not input_json["alternativeUrl"] == '':
                    url = input_json["alternativeUrl"]
                else:
                    url = input_json["url"]
            else:
                url = input_json["url"]

        if not os.path.exists("Data"):
            os.makedirs("Data")

        # Defile subpaths
        video_path = "Data/" + entity + "_" + simmoid + "/"
        frames_path = video_path + "Frames/"
        masks_path = video_path + "Masks/"
        proc_frames_path = video_path + "Proc_Frames/"
        pst_frame_path = video_path + "PST_Frames/"
        only_filename = simmoid + ".mp4"
        save_filename = video_path + only_filename  # This is the path/filename where the downloaded video will be saved  # ToDo: Folders

        # Create directory to save video if it does not exist
        if not os.path.exists(video_path):
            os.makedirs(video_path)

        # Create directory to save masks if it does not exist
        if not os.path.exists(masks_path):
            os.makedirs(masks_path)

        # Create directory to save processed frames if it does not exist
        if not os.path.exists(proc_frames_path):
            os.makedirs(proc_frames_path)

        # Create directory to save video frames
        if not os.path.exists(frames_path) or not exists(video_path+"avg_blurriness.txt"):
            os.makedirs(frames_path, exist_ok=True)

            print("[INFO] Video will be downloaded from ", url)
            logging.info("Video will be downloaded from ", url)

            # Count time to download video and split it in frames
            t1 = time.time()

            # --------------------> Download video
            WebFunctions.download_from_url(url, video_path, only_filename)  #TODO: uncommend this once server is up

            # TODO: commend this ---------------
            online_video = requests.get(url)
            print(online_video)
            logging.info(online_video)

            if online_video.status_code == 200:
                open(save_filename, 'wb').write(online_video.content)
            # TODO: ------------------------------

            video_cap = cv2.VideoCapture(save_filename)
            print("[INFO] Loaded video: ", simmoid + ".mp4")
            logging.info(" Loaded video: ", simmoid + ".mp4")

            # Read and save every video frame
            print("[INFO] Splitting video in frames...")
            logging.info("Splitting video in frames...")
            blur_idx = 0  # number of frames with mean >= 0
            blur_avg = 0
            frame_idx = 0
            while video_cap.isOpened():
                ret, frame = video_cap.read()
                # if frame is read correctly ret is True
                if not ret:
                    print("[INFO] Can't get another frame. Exiting ...")
                    logging.info("Can't get another frame. Exiting ...")
                    break
                else:
                    cv2.imwrite(frames_path + "frame_" + str(frame_idx) + ".jpg", frame)
                    frame_idx = frame_idx + 1

                    # Detect blurry frames, get the avg mean value to set threshold
                    frame_resized = imutils.resize(frame, width=500)
                    gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

                    # apply our blur detector using the FFT
                    (mean, blurry) = detect_blur_fft(gray_frame, size=60, thresh=18)

                    if mean >= 0:
                        blur_avg = blur_avg + mean
                        blur_idx = blur_idx + 1

            video_cap.release()
            print("[INFO] Number of frames: {}, Frames are saved in {}".format(frame_idx - 1, frames_path))
            logging.info("Number of frames: {}, Frames are saved in {}".format(frame_idx - 1, frames_path))
            blur_avg = blur_avg/(blur_idx)

            # Write this in a txt file
            with open(video_path+"avg_blurriness.txt", "w") as blur_file:
                blur_file.write(str(blur_avg))

            t2 = time.time()
            print(t2 - t1, " sec")
            logging.info(t2 - t1, " sec")
        else:
            with open(video_path+"avg_blurriness.txt", "r") as blur_file:
                blur_avg = float(blur_file.readline())

        print("[INFO] Average blurriness in video is ", blur_avg)
        logging.info("Average blurriness in video is ", blur_avg)

        # --------------------------> Shot Detection
        print("[INFO] Splitting video into shots...")
        logging.info("Splitting video into shots...")
        # Count time for video shot detection
        t1 = time.time()
        subprocess.run("python TransNetV2-master/inference/transnetv2.py " + save_filename)

    #     # predictors.shot_detection(save_filename)
    #     # K.clear_session()  # TODO!!!
    #
        # Save shot detection info in lists to use it
        start_frame = []
        end_frame = []

        # Read txt with shot info
        shot_filename = save_filename+".scenes.txt"
        shot_file = open(shot_filename, "r")
        shots_num = 0
        for line in shot_file:
            start_frame.append(line.split()[0])
            end_frame.append(line.split()[-1])
            shots_num = shots_num + 1
        print("[INFO] Shot boundary detection finished!")
        logging.info("Shot boundary detection finished!")
        t2 = time.time()
        print(t2 - t1, " sec")
        logging.info(t2 - t1, " sec")

        # ------------- Scene Recognition (for every shot) --------------#
        #                              &                                 #
        # ------------- Emergency Detection (for every shot) ------------#

        # Using keyframes to get a result that characterizes the whole video shot
        # ToDo: Extract keyframes per video shot and analyze only them!!!

        # Create the needed dictionaries to save extracted info   # TODO: add json output for geoserver! + ProjectID?

        # For KB - Information only per video shot
        KB_JSON = {}
        KB_JSON["header"] = {"timestamp": str(datetime.now(tz=None)), "sender": "Visual Analysis", "entity": entity, "simmoid": simmoid, "project_id": project_id}  #ToDo
        KB_JSON["shotInfo"] = []

        # For 3D Rec - Information per video shot for SR + all processed frames
        Rec_JSON = {}
        Rec_JSON["header"] = {"timestamp": str(datetime.now(tz=None)), "sender": "Visual Analysis", "entity": entity,
                              "simmoid": simmoid, "project_id": project_id}  # ToDo test with project_id too!!
        Rec_JSON["shotInfo"] = []

        # Information to keep in my MongoDB (VisualsOutput collection) - Full analysis info (per shot for SR, per frame BOL)
        full_JSON  = {}
        full_JSON["header"] = {"timestamp": str(datetime.now(tz=None)), "entity": entity, "simmoid": simmoid}
        full_JSON["shotInfo"] = []

        video_info_dict = {}
        video_info_dict["simmoid"] = simmoid

        print("[INFO] ----> Analysing video, please wait...")
        logging.info("----> Analysing video, please wait...")

        # load model for SR
        SR_model = load_model('Trained_models/SR/xr4drama_places_model_ep5_bs32.h5')
        print("[INFO] SR model is successfully loaded!")
        logging.info("SR model is successfully loaded!")

        # load model for EmC
        EmC_model = load_model('Trained_models/EmC/vgg16_places2_lastconv_19ep_1e-03lr_MME2017.pkl')
        print("[INFO] EmC model is successfully loaded!")
        logging.info("EmC model is successfully loaded!")

        if pst:
            # load model for PST
            # train_tfrec = "PST/tfrecords/train.tfrec"
            # val_tfrec = "PST/tfrecords/val.tfrec"

            model = WCT2(image_size=None, lr=1e-4, gram_loss_weight=1.0)
            model.wct.load_weights(model.checkpoint_path)
            # model.train(train_tfrec, epochs=10, batch_size=8)

            model.wct.save_weights(model.checkpoint_path)

            print("[INFO] PST model is successfully loaded!")
            logging.info("PST model is successfully loaded!")

        # load model for BOL
        semantic_segmentation_model = semantic_segmentation()
        semantic_segmentation_model.load_ade20k_model("Trained_models/BOL/Pixellib/deeplabv3_xception65_ade20k.h5")
        print("[INFO] BOL model is successfully loaded!")
        logging.info("BOL model is successfully loaded!")

        for shot_idx in range(0, shots_num):

            # # Create a separate folder for the processed frames of each video shot
            # if not os.path.exists(proc_frames_path + "shot" + str(shot_idx)):
            #     os.makedirs(proc_frames_path + "shot" + str(shot_idx))

            # Save extracted info (per video shot) - for KB
            shot_info_dict_KB = {}
            shot_info_dict_KB["shotIdx"] = shot_idx
            shot_info_dict_KB["startFrame"] = start_frame[shot_idx]
            shot_info_dict_KB["endFrame"] = end_frame[shot_idx]
            # shot_info_dict_KB["area"] = []
            # shot_info_dict_KB["areaProb"] = []
            # shot_info_dict_KB["outdoor"] = []
            # shot_info_dict_KB["emergencyType"] = []
            # shot_info_dict_KB["emergencyProb"] = []
            # shot_info_dict_KB["objectsFound"] = [{"type": "building", "probability": 0.67}, {"type": "traffic light", "probability": 0.44}]  # TODO: add actual objects
            shot_info_dict_KB["objectsFound"] = []
            shot_info_dict_KB["peopleInDanger"] = 0
            shot_info_dict_KB["vehiclesInDanger"] = 0
            shot_info_dict_KB["riverOvertop"] = False

            # Save extracted info (per video shot) - for 3D Rec
            shot_info_dict_3D_rec = {}
            shot_info_dict_3D_rec["shotIdx"] = shot_idx
            shot_info_dict_3D_rec["startFrame"] = start_frame[shot_idx]
            shot_info_dict_3D_rec["endFrame"] = end_frame[shot_idx]
            shot_info_dict_3D_rec["frameInfo"] = []

            frame_info_dict_3D_rec = {}
            # frame_info_dict_3D_rec["frameNum"] = 0
            # frame_info_dict_3D_rec["procFrameUrl"] = ""

            # For the output to 3d reconstruction
            blur_classes = ["person", "car", "bus", "truck", "minibike", "bicycle", "van", "animal"]
            remove_classes = ["sky", "sea", "signboard"]
            # keep_classes = ["wall", "building", "road", "sidewalk", "house", "column", "skyscraper", "path", "stairs", \
            #                   "stairway", "bridge", "tower", "fountain", "sculpture"]
            #
            # keep_classes_2 = ["wall", "building", "floor", "road", "grass", "sidewalk", "earth", "door", "mountain", \
            #                 "house", "field", "fence", "river", "rock", "column", "skyscraper", "path", "stairs", \
            #                 "stairway", "bridge", "hill", "bench", "tower", "land", "escalator", "fountain", "swimming", \
            #                 "sculpture"]

            print("\n\n[INFO] ----> Analyzing shot ", shot_idx)
            logging.info("----> Analyzing shot ", shot_idx)
            n = 4  # get SR result per n frames
            frames = []
            scenes = []
            sr_probs = []
            are_outdoor = []
            emergencies = []
            emc_probs = []
            frame_counter = 0 # how many frames were analysed
            frames_for_rec_num = 0 # how many frames were automatically selected to be sent to 3D reconstruction service
            for frame_idx in range(int(start_frame[shot_idx])+5, int(end_frame[shot_idx])-5, n):  # +5 frames from the start frame of the video shot to avoid shot transition artifacts

                # Add frame number for the output to 3d reconstruction
                frame_info_dict_3D_rec["frameNum"] = frame_idx
                # frame_info_dict_3D_rec["procFrameUrl"] = ""

                frames.append(frame_idx)
                # print("Number of frame analyzed: ", frame_idx)
                # Get frame
                frame = cv2.imread(frames_path + "frame_" + str(frame_idx) + ".jpg")

                # Run SR model
                scene, sr_prob, is_outdoor = predictors.SR_model(frame, SR_model)
                scenes.append(scene)
                sr_probs.append(sr_prob)
                are_outdoor.append(is_outdoor)
                print("Scene: {}, Probability: {}".format(scene, sr_prob))
                logging.info("Scene: {}, Probability: {}".format(scene, sr_prob))

                # Run EmC model
                emergency, emc_prob = predictors.EmC_model(frame, EmC_model)
                emergencies.append(emergency)
                emc_probs.append(emc_prob)

                # Check if we will use PST or not
                if pst:
                    os.makedirs(pst_frame_path, exist_ok=True)
                    # rst = None
                    rst = 64 * 12  # (896)
                    # test_id = np.random.randint(1, 60)
                    # test_id = 20

                    # start = datetime.datetime.now()
                    # content = get_local_img("./examples/input/tar55.png".format(test_id), rst)
                    content = get_local_img(frames_path + "frame_" + str(frame_idx) + ".jpg", rst)
                    style = get_local_img("PST/examples/style/tar44.png", rst)

                    output = model.transfer(tf.cast(content, tf.float32), tf.cast(style, tf.float32), 0.8)

                    plt.imsave(pst_frame_path + "frame_" + str(frame_idx) + ".jpg", output[0] / 255.0)

                    # Run BOL model
                    segvalues, objects_masks, image_overlay = semantic_segmentation_model.segmentAsAde20k(
                        pst_frame_path + "frame_" + str(frame_idx) + ".jpg", overlay=False, extract_segmented_objects=True,
                        output_image_name=masks_path + "frame_" + str(frame_idx) + "_mask.png")

                else:

                    # Run BOL model
                    segvalues, objects_masks, image_overlay = semantic_segmentation_model.segmentAsAde20k(frames_path + "frame_" + str(frame_idx) + ".jpg", overlay=False, extract_segmented_objects=True,
                                                                                        output_image_name=masks_path + "frame_" + str(frame_idx) + "_mask.png")

                print(frames_path + "frame_" + str(frame_idx) + ".jpg")
                logging.info(frames_path + "frame_" + str(frame_idx) + ".jpg")
                print(segvalues['class_names'])
                logging.info(segvalues['class_names'])
                print(segvalues['ratios'])
                logging.info(segvalues['ratios'])

                # Upload mask to file storage
                url = "http://xr4drama.iti.gr:5002/fileUpload/masks"
                # Open the image in read-only format.
                file = {'file': open(masks_path + "frame_" + str(frame_idx) + "_mask.png", 'rb')}
                uploaded_file_url = WebFunctions.send_post(url, file, False)
                print("Mask url: ", uploaded_file_url)
                logging.info("Mask url: ", uploaded_file_url)

                building_class = False
                building_ratio = 0
                wall_class = False
                wall_ratio = 0

                if len(shot_info_dict_KB['objectsFound']) > 0:
                    for o in range(0, len(segvalues['class_names'])):
                        found = 0  # flag to see if label was saved so that we have it once for the shot
                        for dict_idx in range(len(shot_info_dict_KB['objectsFound'])):
                            my_dict = shot_info_dict_KB['objectsFound'][dict_idx]

                            if segvalues['class_names'][o] == my_dict['type']:
                                found = 1
                                # check if the current probability is greater than the saved one
                                if my_dict['probability'] < segvalues['ratios'][o]:
                                    shot_info_dict_KB['objectsFound'][dict_idx]["probability"] = segvalues['ratios'][o]

                        if found == 0 and segvalues['class_names'][o] in predictors.wanted_ade20k_labels and segvalues['ratios'][o] > 1:  # 1/100 = 0.01 threshold of ratio to take into account the class
                            objects_dict = {}
                            objects_dict['type'] = segvalues['class_names'][o]
                            objects_dict['probability'] = segvalues['ratios'][o] / 100
                            # objects_info.append(objects_dict)
                            shot_info_dict_KB['objectsFound'].append(objects_dict)

                        # Check if there is "building" or "wall" and get their ratios
                        if segvalues['class_names'][o] == "building":
                            building_class = True
                            building_ratio = segvalues['ratios'][o]   #0-100
                        elif segvalues['class_names'][o] == "wall":
                            wall_class = True
                            wall_ratio = segvalues['ratios'][o]   #0-100

                else:
                    for o in range(0, len(segvalues['class_names'])):
                        if segvalues['class_names'][o] in predictors.wanted_ade20k_labels and segvalues['ratios'][o] > 1:
                            objects_dict = {}
                            objects_dict['type'] = segvalues['class_names'][o]
                            objects_dict['probability'] = segvalues['ratios'][o] / 100
                            shot_info_dict_KB['objectsFound'].append(objects_dict)

                        # Check if there is "building" or "wall" and get their ratios
                        if segvalues['class_names'][o] == "building":
                            building_class = True
                            building_ratio = segvalues['ratios'][o]  # 0-100
                        elif segvalues['class_names'][o] == "wall":
                            wall_class = True
                            wall_ratio = segvalues['ratios'][o]  # 0-100

                frame_counter = frame_counter + 1

                # Check if we should extract output for the 3D reconstruction service
                if is_outdoor and sr_prob >= 9.5 and scene != "bar" and scene != "restaurant" and \
                        ((building_class and building_ratio >= 8) or (wall_class and wall_ratio >= 8)):

                    # Blurriness check
                    frame_resized = imutils.resize(frame, width=500)
                    gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

                    # apply our blur detector using the FFT
                    (mean, blurry) = detect_blur_fft(gray_frame, size=60,
                        thresh=blur_avg-2)

                    print("---> Detecting bluriness - Mean: ", mean)
                    logging.info("---> Detecting bluriness - Mean: ", mean)

                    if blurry:
                        print("^^^^^Blurry frame detected!^^^^^")
                        logging.info("^^^^^Blurry frame detected!^^^^^")
                    else:
                        # Current frame was selected to be sent to 3D reconstruction service
                        frames_for_rec_num = frames_for_rec_num + 1

                        # Create output image for 3D reconstruction
                        h, w, n = frame.shape

                        blur_mask = np.zeros(shape=(h, w), dtype=np.uint8)
                        remove_mask = np.zeros(shape=(h, w), dtype=np.uint8)

                        for i in range(0, len(objects_masks)):
                            print(objects_masks[i]['class_name'])
                            logging.info(objects_masks[i]['class_name'])

                            if objects_masks[i]['class_name'] in remove_classes:

                                mask = objects_masks[i]['masks']
                                mask = mask.astype(np.uint8)
                                mask[mask == 1] = 255

                                new_mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

                                # Add this mask to remove_mask
                                remove_mask = remove_mask + new_mask

                            # if there is a great coverage of "building", remove "mountain"
                            elif objects_masks[i]['class_name'] == "mountain" and building_ratio >= 50:

                                mask = objects_masks[i]['masks']
                                mask = mask.astype(np.uint8)
                                mask[mask == 1] = 255

                                new_mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

                                # Add this mask to remove_mask
                                remove_mask = remove_mask + new_mask

                            if objects_masks[i]['class_name'] in blur_classes:
                                mask = objects_masks[i]['masks']
                                mask = mask.astype(np.uint8)
                                mask[mask == 1] = 255

                                new_mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

                                # Add this mask to blur_mask
                                blur_mask = blur_mask + new_mask

                        # Blur specific objects
                        img = frame.copy()
                        blur_img = cv2.blur(frame, (21, 21), 0)
                        blur_mask_3d = cv2.merge((blur_mask, blur_mask, blur_mask))
                        out = np.where(blur_mask_3d, blur_img, img)

                        # Keep only the foreground in image
                        inv_remove_mask = cv2.bitwise_not(remove_mask)

                        # Remove noise from mask with dilation and erosion
                        # Taking a matrix of size 5 as the kernel
                        kernel = np.ones((5, 5), np.uint8)

                        # The first parameter is the original image,
                        # kernel is the matrix with which image is
                        # convolved and third parameter is the number
                        # of iterations, which will determine how much
                        # you want to erode/dilate a given image.
                        inv_remove_mask_eros = cv2.erode(inv_remove_mask, kernel, iterations=10)
                        inv_remove_mask_denoised = cv2.dilate(inv_remove_mask_eros, kernel, iterations=10)

                        # for r in range(0, 5):
                        #     kernel = np.ones((7, 7), np.uint8)   # TODO: Best kernel size?
                        #     inv_remove_mask_denoised = cv2.morphologyEx(inv_remove_mask, cv2.MORPH_OPEN, kernel)

                        # Combine blurring & masking
                        out2 = cv2.bitwise_and(out, out, mask=inv_remove_mask_denoised)

                        # Create a separate folder for the processed frames of each video shot
                        os.makedirs(proc_frames_path + "shot" + str(shot_idx), exist_ok=True)

                        cv2.imwrite(proc_frames_path + "shot" + str(shot_idx) + "/frame_" + str(frame_idx) + "_processed.jpg", out2)

                        # Upload processed frame to file storage
                        url = "http://xr4drama.iti.gr:5002/fileUpload/processed_images"
                        # Open the image in read-only format.
                        file = {'file': open(proc_frames_path + "shot" + str(shot_idx) + "/frame_" + str(frame_idx) + "_processed.jpg", 'rb')}
                        uploaded_file_url = WebFunctions.send_post(url, file, False)
                        print("Processed frame url: ", uploaded_file_url)
                        logging.info("Processed frame url: ", uploaded_file_url)
                        frame_info_dict_3D_rec["procFrameUrl"] = uploaded_file_url

                        # Add extracted information for this frame to the 3D Rec dictionary
                        shot_info_dict_3D_rec["frameInfo"].append(frame_info_dict_3D_rec.copy())

            # Dominant Scene - Majority voting
            c = Counter(scenes)
            dominant_scene = c.most_common(1)  # finds most common scene detected in the frames of the analyzed shot

            if len(dominant_scene) == 0:
                dominant_scene.append(('none', 0))

            shot_info_dict_KB["area"] = dominant_scene[0][0]
            # shot_info_dict_Rec["area"] = dominant_scene[0][0]
            # shot_json["shots_info"][shot_idx]["outdoor"] = True
            # shot_json["shots_info"][shot_idx]["scene_recognized"] = dominant_scene[0][0]
            print("[INFO] Scene Recognized in shot {} : {} - ({}/{})".format(shot_idx, dominant_scene[0][0], dominant_scene[0][1], frame_counter))
            logging.info("Scene Recognized in shot {} : {} - ({}/{})".format(shot_idx, dominant_scene[0][0], dominant_scene[0][1], frame_counter))

            # Avg probability of Dominant Scene
            # print(dominant_scene[0][0])
            indices = [i for i, s in enumerate(scenes) if dominant_scene[0][0] in s]
            # print(indices)
            sum_sr_probs = 0
            for idx in indices:
                sum_sr_probs = sum_sr_probs + float(sr_probs[idx])

            if dominant_scene[0][1] == 0:
                avg_sr_prob = 0
            else:
                avg_sr_prob = sum_sr_probs / dominant_scene[0][1]
                shot_info_dict_KB["areaProb"] = avg_sr_prob/100
                # shot_info_dict_Rec["areaProb"] = avg_sr_prob / 100
                print("----- Avg SR Probability: {}%".format(avg_sr_prob))
                logging.info("----- Avg SR Probability: {}%".format(avg_sr_prob))
                # shot_json["shots_info"][shot_idx]["scene_prob"] = avg_sr_prob

            # Characterize analyzed scene as "outdoors" or "indoors"
            c = Counter(are_outdoor)
            outdoor_instances = c.most_common(1)

            if len(outdoor_instances) == 0:
                outdoor_instances.append(('none', 0))

            outdoor = outdoor_instances[0][0]
            shot_info_dict_KB["outdoor"] = outdoor
            # shot_info_dict_Rec["outdoor"] = outdoor
            if outdoor:
                print("----- Scene is characterized as Outdoor.")
                logging.info("----- Scene is characterized as Outdoor.")
            else:
                print("----- Scene is characterized as Indoor.")
                logging.info("----- Scene is characterized as Indoor.")

            # Dominant emergency situation and avg probability  ToDo: It now outputs dummy results!! Add beAware model for EmC
            c = Counter(emergencies)
            dominant_emergency = c.most_common(1)

            if len(dominant_emergency) == 0:
                dominant_emergency.append(('none', 0))

            shot_info_dict_KB["emergencyType"] = dominant_emergency[0][0]
            print("[INFO] Emergency type in shot: ", dominant_emergency[0][0])
            logging.info("Emergency type in shot: ", dominant_emergency[0][0])

            # Avg probability of Dominant Emergency Type
            indices = [i for i, s in enumerate(emergencies) if dominant_emergency[0][0] in s]
            # print(indices)
            sum_emc_probs = 0
            for idx in indices:
                sum_emc_probs = sum_emc_probs + float(emc_probs[idx])

            if dominant_emergency[0][1] == 0:
                avg_emc_prob = 0
            else:
                avg_emc_prob = sum_emc_probs / dominant_emergency[0][1]
                shot_info_dict_KB["emergencyProb"] = avg_emc_prob
                # shot_json["shots_info"][shot_idx]["emergency_type"] = dominant_emergency[0][0]
                # shot_json["shots_info"][shot_idx]["emergency_prob"] = avg_emc_prob
                print('----- Avg Emergency Probability: {}%'.format(avg_emc_prob))
                logging.info('----- Avg Emergency Probability: {}%'.format(avg_emc_prob))

            # Finalise the output JSON for KB
            # print(shot_info_dict_KB)
            #
            KB_JSON["shotInfo"].append(shot_info_dict_KB)

            print(KB_JSON["shotInfo"][shot_idx])
            logging.info(KB_JSON["shotInfo"][shot_idx])

            if frames_for_rec_num >= 5:
                # Update output JSON for 3D Reconstruction
                Rec_JSON["shotInfo"].append(shot_info_dict_3D_rec)

                print(Rec_JSON["shotInfo"][-1])
                logging.info(Rec_JSON["shotInfo"][-1])

        # Save KB_JSON file
        with open(video_path + entity + '_' + simmoid + "_" + 'KB.json', 'w') as f:
            json.dump(KB_JSON, f)

        # Save output for KB (maybe this will be inside the analysis function)
        db.ImagesDB.VisualsOutputKB.insert_one(KB_JSON)
        # db.XR4D_Visual_Analysis_DB.VisualsOutputKB.insert_one(KB_JSON)

        if sendtoKB:
            print("\n[INFO] Sending output to KB...")
            logging.info("\nSending output to KB...")

            # /population/VISUAL_ANALYSIS

        # Save Rec_JSON file
        with open(video_path + entity + '_' + simmoid + "_" + "Rec.json", 'w') as f:
            json.dump(Rec_JSON, f)

        # Save output for KB (maybe this will be inside the analysis function)
        db.ImagesDB.VisualsOutput3D.insert_one(Rec_JSON)
        # db.XR4D_Visual_Analysis_DB.VisualsOutput3D.insert_one(Rec_JSON)

        if sendtoRec:
            print("\n[INFO] Sending output to 3D Reconstruction...")
            logging.info("\nSending output to 3D Reconstruction...")

            # Send Rec_JSON to the 3D Reconstruction service
            base_url = "https://baremetal.up2metric.com/"

            rec_file = open(video_path + entity + '_' + simmoid + "_" + "Rec.json", "r")
            payload = json.load(rec_file)

            headers = {
                'api_key': '2gLTq6KTBFCevaXlQ7lEva-pylNfSqy6MUQpr1n26BGDOqut8K7en4IuxV6S2r7KJ9SuWOyvD7oihBy7pqZKAA',
                # 'oCxNWXZYI5ksEno0Bb8Nf_Os4vgaqf0R07MwElmbKR-xiKpZLAphlcLJW76IYKYITYn_cLTiE5xGYeeIocd4Fw',
                'Content-Type': 'application/json'
            }

            response = requests.post(base_url + "jobs/json", headers=headers, json=payload)
            print(response.text)
            logging.info(response.text)
            print(response.status_code)
            logging.info(response.status_code)

    #### ====== Create Log File ====== ####
    # Get date and time that current analysis ends
    moment = str(datetime.now(tz=None))

    # Get current log data
    with open("logfilename.log", "r") as f:
        logs = f.read()

    # Create current log dictionairy
    new_log_data = {}
    new_log_data["moment"] = moment
    new_log_data["message"] = logs

    # Get all logs until now

    # Check if log file exists
    file_exists = exists("full_log_file.json")
    if not file_exists:
        open("full_log_file.json", "w")

    # Check if log file is empty
    if os.stat("full_log_file.json").st_size == 0:
        print('Log File is empty')

        full_log_list = []
        full_log_list.append(new_log_data)

        with open("full_log_file.json", "w") as file:
            json.dump(full_log_list, file)

    else:
        print('Log File is not empty')

        with open("full_log_file.json", "r") as file_read:
            past_log_data = json.load(file_read)

            past_log_data.append(new_log_data)

            with open("full_log_file.json", "w") as file_write:
                json.dump(past_log_data, file_write)




