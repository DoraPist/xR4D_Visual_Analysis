################################################################
# This script contains all the predictors used in the visual analysis service .
# 10/06/2021 - Theodora Pistola
################################################################

import subprocess
import cv2
import numpy as np
from keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input
from keras import backend as K
import pandas as pd
import tensorflow as tf
# from deeplab_resnet import DeepLabResNetModel, ImageReader, decode_labels
from PIL import Image
import os
from web_functions import WebFunctions
import time
import json
from os import listdir
from os.path import isfile, join
from BOL_files import deeplab_model
from BOL_files.utils import preprocessing
from BOL_files.utils import dataset_util
import matplotlib.pyplot as plt

# # labels' colors
# MAP_palette = {(0, 0, 0): 0,  # 0 other
#                (196, 196, 196): 1,  # 1 barrier--curb
#                (190, 153, 153): 2,  # 2 barrier--fence
#                (180, 165, 180): 3,  # 3 barrier--guard-rail
#                (90, 120, 150): 4,  # 4 barrier--other-barrier
#                (102, 102, 156): 5,  # 5 barrier--wall
#                (128, 64, 255): 6,  # 6 flat--bike-lane
#                (140, 140, 200): 7,  # 7 flat--crosswalk-plain
#                (170, 170, 170): 8,  # 8 flat--curb-cut
#                (250, 170, 160): 9,  # 9 flat--parking
#                (96, 96, 96): 10,  # 10 flat--pedestrian-area
#                (230, 150, 140): 11,  # 11 flat--rail-track
#                (128, 64, 128): 12,  # 12 flat--road
#                (110, 110, 110): 13,  # 13 flat--service-lane
#                (244, 35, 232): 14,  # 14 flat--sidewalk
#                (150, 100, 100): 15,  # 15 structure--bridge
#                (70, 70, 70): 16,  # 16 structure--building
#                (150, 120, 90): 17}  # 17 structure--tunnel
#
# # label colors for the common classes full dataset for Facade Segmentation
# FS_palette = {(0, 0, 0): 0,  # 0 other
#               (255, 10, 0): 1,  # 1 facade
#               (0, 10, 255): 2,  # 2 window
#               (0, 236, 0): 3}  # 3 door

######################################################################
# TensorFlow wizardry
#config = tf.ConfigProto()

# Don't pre-allocate memory; allocate as-needed
# config.gpu_options.allow_growth = True

# Only allow a total of half the GPU memory to be allocated
# config.gpu_options.per_process_gpu_memory_fraction = 0.8

# Create a session with the above options specified.
# K.tensorflow_backend.set_session(tf.Session(config=config))

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
MAP_NUM_CLASSES = 18
FS_NUM_CLASSES = 4
MAP_TRAINED_MODEL = "Trained_models/BOL/STBL_models/model.ckpt-160000"
FS_TRAINED_MODEL = "Trained_models/BOL/STBL_models/STBL_models/cmp_full_model.ckpt-70000"

# # labels' names
# MAP_labels = {0: "other",
#               1: "barrier--curb",
#               2: "barrier--fence",
#               3: "barrier--guard-rail",
#               4: "barrier--other-barrier",
#               5: "barrier--wall",
#               6: "flat--bike-lane",
#               7: "flat--crosswalk-plain",
#               8: "flat--curb-cut",
#               9: "flat--parking",
#               10: "flat--pedestrian-area",
#               11: "flat--rail-track",
#               12: "flat--road",
#               13: "flat--service-lane",
#               14: "flat--sidewalk",
#               15: "structure--bridge",
#               16: "structure--building",
#               17: "structure--tunnel"}
#
# FS_labels = {0: "other",
#              1: "facade/wall",
#              2: "window",
#              3: "door"}


# def shot_detection(video_path):
#     subprocess.run("python TransNetV2-master/inference/transnetv2.py "+video_path)

sess = tf.compat.v1.Session()
_BOL_NUM_CLASSES = 19


BOL_labels = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
              "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle",
              "bicycle"]

wanted_ade20k_labels = ['wall', 'building', 'sky', 'floor;flooring', 'tree', 'ceiling', 'road', \
                 'grass', 'sidewalk;pavement', \
                 'person', 'earth', 'door', 'table', \
                 'mountain', 'plant',  \
                 'chair', 'car', 'water', \
                 'house', 'sea', 'field', 'fence;fencing',\
                 'rock;stone', 'lamp', 'railing;rail', \
                 'base;pedestal;stand', 'column;pillar', 'signboard;sign', \
                 'sand', 'skyscraper', \
                 'grandstand;covered;stand', 'path', 'stairs;steps', 'runway', 'case;display;case;showcase;vitrine', \
                 'pool;table;billiard;table;snooker;table', 'screen;door;screen', 'stairway;staircase',\
                 'river', 'bridge;span', 'blind;screen', 'coffee;table;cocktail;table', \
                 'hill', 'bench', 'countertop', \
                 'palm;palm;tree', \
                 'boat', 'bar', 'arcade;machine', 'hovel;hut;hutch;shack;shanty', \
                 'bus', \
                 'light;light;source', 'truck', 'tower', \
                 'awning;sunshade;sunblind', 'streetlight;street;lamp', 'booth;cubicle;stall;kiosk', \
                 'airplane;aeroplane;plane', 'dirt;track', 'pole', \
                 'land;ground;soil', 'bannister;banister;balustrade;balusters;handrail', \
                 'escalator;moving;staircase;moving;stairway', \
                 'poster;posting;placard;notice;bill;card', 'stage', 'van', 'ship', \
                 'fountain', 'canopy', \
                 'swimming;pool;swimming;bath;natatorium',\
                 'waterfall;falls', 'tent;collapsible;shelter', \
                 'minibike;motorbike', 'step;stair', 'tank;storage;tank', \
                 'trade', \
                 'animal;', 'bicycle', 'lake', \
                 'sculpture', 'traffic;light;traffic;signal;stoplight',\
                 'ashcan;trash;can;garbage;can;wastebin;ash;bin;ash-bin;ashbin;dustbin;trash;barrel;trash;bin', \
                 'pier;wharf;wharfage;dock', \
                 'clock', 'flag']


def SR_model(frame, model1):

    # with tf.compat.v1.variable_scope("keras_graph"):
    #     # load model for SR   # ToDo: way not load model everytime?
    #     model1 = load_model('Trained_models/SR/my_newmodel_places_kle_sept.h5')
    #     print("[INFO] SR model is successfully loaded!")

    # load selected 152 categories of Places
    # classes_152 = np.array(pd.read_csv('Trained_models/SR/categories_152.csv', header=None))
    classes_99 = np.array(pd.read_csv('Trained_models/SR/categories_99.csv', header=None))

    img = cv2.resize(frame, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img, mode='caffe')

    # predict the scene class
    prediction = model1.predict(img)

    # del model1
    # K.clear_session()

    c = np.amax(prediction)  # maximum value - max probability
    d = np.argmax(prediction)  # class number

    # check class and indoor/outdoor
    pred_class = classes_99[d, 0]
    prob = round(c * 100, 2)
    # print("[SR] \nPredicted class = {}\nProbability = {:.2f}%".format(pred_class, round(c * 100, 2)))

    # load the categories of Places that are outdoors
    # outdoor_classes = np.array(pd.read_csv('Trained_models/SR/categories_outdoor.csv', header=None))
    # if d in outdoor_classes[:, 1]:
    #     outdoor = True
    # else:
    #     outdoor = False
    if pred_class == "indoor":
        outdoor = False
    else:
        outdoor = True

    # return "beach", "0.79", True
    return pred_class, prob, outdoor

def EmC_decode_predictions(preds):
    class_index = json.load(open("Trained_models/EmC/ivmsp_index.json"))
    sorted_preds = np.argsort(-preds[0])
    results = []
    for idx in sorted_preds:
        results += [[class_index[str(idx)], preds[0][idx]]]
    return results


def EmC_model(frame, model1):

    # Preprocess image
    img = cv2.resize(frame, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img, mode='caffe')

    emc_pred = model1.predict(img)
    pred_class = EmC_decode_predictions(emc_pred)[0][0].lower()
    prob = emc_pred.max()
    return pred_class, prob
    # return "none", "1"


def create_txt_for_test_images(my_path):
    with open(my_path + "frames_list.txt", "w") as file:

        onlyfiles = [f for f in listdir(my_path) if isfile(join(my_path, f))]
        for i in onlyfiles:
            if not "txt" in i:
                file.write(i + '\n')
                # print(i)

                # Resize image if needed
                # Load image
                im = cv2.imread(my_path + i)
                h, w, _ = im.shape
                # print("h=" + str(h) + " w=" + str(w))
                scale_percent = 60  # percent of the original image

                # RESIZES IMAGES !!! TODO: KEEP A COPY OF THE ORIGINAL ONES FIRST!!!
                if h >= 1024 or w >= 1024:
                    new_h = int(h * scale_percent / 100)
                    new_w = int(w * scale_percent / 100)
                    dim = (new_w, new_h)

                    im_res = cv2.resize(im, dim, interpolation=cv2.INTER_AREA)

                    cv2.imwrite(my_path + i, im_res)


def calculate_probabilities_per_label(pixel_probabilities):
    # compute average probability / class
    pixels_class = np.zeros((1, _BOL_NUM_CLASSES))  # save the number of pixels belong to each class
    sum_prob_class = np.zeros((1, _BOL_NUM_CLASSES))  # save the sum of probabilities for each class
    avg_prob_class = np.zeros(
        (1, _BOL_NUM_CLASSES))  # save average probability for each class (normalize it in the end 0-1)
    pixel_counter = 0
    sh = pixel_probabilities.shape
    for i in range(0, sh[0]):
        for j in range(0, sh[1]):
            out = pixel_probabilities[i][j]
            # print((output2[0][i][j]).shape)
            label_selected = np.argmax(out)
            max_probability = np.max(out)
            # print(label_selected)
            # print("pixel No{}".format(pixel_counter)
            pixels_class[0, label_selected] += 1
            sum_prob_class[0, label_selected] += max_probability
            pixel_counter += 1

    # Calculate average probability values for each label
    for c in range(0, _BOL_NUM_CLASSES):
        if pixels_class[0, c] != 0:
            avg_prob_class[0, c] = sum_prob_class[0, c] / pixels_class[0, c]

    return avg_prob_class


def segmentation(semantic_segmentation_model, image_file, output_dir):


    segvalues, segoverlay = semantic_segmentation_model.segmentAsAde20k(image_file, overlay=False,
                                                                        output_image_name=output_dir)

    # Using the Winograd non-fused algorithms provides a small performance boost.
    # os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    # pred_hooks = None
    #
    # # Create dictionary to save all the results
    # body = []
    #
    # # load segmentation model
    # model = tf.estimator.Estimator(
    #     model_fn=deeplab_model.deeplabv3_plus_model_fn,
    #     model_dir="Trained_models/BOL/CityScapes_model",
    #     params={
    #         'output_stride': 8,
    #         'batch_size': 1,  # Batch size must be 1 because the images' size may differ
    #         'base_architecture': 'resnet_v2_101',
    #         'pre_trained_model': None,
    #         'batch_norm_decay': None,
    #         'num_classes': 19,
    #     })
    #
    # examples = dataset_util.read_examples_list(images_list)
    # image_files = [os.path.join(images_dir, filename) for filename in examples]
    #
    # # run segmentation model on image
    # predictions = model.predict(
    #     input_fn=lambda: preprocessing.eval_input_fn(image_files),
    #     hooks=pred_hooks)
    #
    # # create output_dir if it does not exist
    # # output_dir = output_dir
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    #
    # output = []
    # # decode predictions and save segmentation mask for each image
    # for pred_dict, image_path in zip(predictions, image_files):
    #
    #     # t1 = time.time()  # start time for 1 image
    #     # image_name = os.path.basename(image_path)
    #
    #     # # create dictionary
    #     # output = []
    #     # full_json['id'] = image_name
    #     # BOL_info = {}
    #     # BOL_info['type'] = []
    #     # BOL_info['probability'] = []
    #     # output['segm_probabilities'] = []
    #     # full_json['segmentation'] = output
    #
    #     image_basename = os.path.splitext(os.path.basename(image_path))[0]
    #     # output['frame'] = image_basename
    #     output_filename = image_basename + '_mask.png'
    #     path_to_output = os.path.join(output_dir, output_filename)
    #
    #     print("generating:", path_to_output)
    #     mask = pred_dict['decoded_labels']
    #     mask = Image.fromarray(mask)
    #     plt.axis('off')
    #     plt.imshow(mask)
    #     plt.savefig(path_to_output, bbox_inches='tight')
    #
    #     # Get probabilities - confidence levels
    #     probabilities = pred_dict['probabilities']
    #     avg_prob_class = calculate_probabilities_per_label(probabilities)
    #     # output['segm_probabilities'] = avg_prob_class
    #
    #     # Keep only the objects with probability over the probability_threshold
    #     probability_threshold = 0.2
    #     # For each class
    #     for idx in range(0, _BOL_NUM_CLASSES):
    #         BOL_info = {}
    #         if avg_prob_class[0, idx] > probability_threshold:
    #             BOL_info["type"] = BOL_labels[idx]
    #             # print(BOL_labels[idx])
    #             BOL_info["probability"] = avg_prob_class[0, idx]
    #             # print(avg_prob_class[0, idx])
    #
    #             # Check if we have already localised this object in this video shot - keep max prob
    #             found = 0
    #             if len(output) > 0:
    #                 for dict_idx in range(len(output)):
    #                     my_dict = output[dict_idx]
    #                     if BOL_labels[idx] in my_dict.values():
    #                         found = 1
    #                         # check if the current probability is greater than the saved one
    #                         if my_dict["probability"] < BOL_info["probability"]:
    #                             output[dict_idx]["probability"] = BOL_info["probability"]
    #
    #             # If it does not exist in the output list put it there
    #             if found == 0:
    #                 output.append(BOL_info)

        # print("BOL_info output", output)

    return output


def ObjD_model(frames_path, masks_path, filename, new_images, height, width):
    # mask = cv2.imread("D:/tpistola/Projects/xR4DRAMA/Visual_Analysis/SR_v1/videos/Corfu/Corfu2/masks/MAP_masks/2552_Mask1.png")
    # objects = []
    # obj_probs = []
    #
    # # Load the semantic segmentation model
    # with tf.Graph().as_default() as net1_graph:
    #
    #     # Create queue coordinator.
    #     coord1 = tf.train.Coordinator()
    #
    #     # Load reader.
    #     with tf.name_scope("create_inputs1"):
    #         reader1 = ImageReader(
    #             frames_path,
    #             filename,
    #             None,  # No defined input size.
    #             False,  # No random scale.
    #             False,  # No random mirror.
    #             255,  # Ignore label
    #             IMG_MEAN,
    #             coord1)
    #         # print("reader queue ", reader1.image_list[0])
    #         image1, label1 = reader1.image, reader1.label
    #         title1 = reader1.queue[0]
    #     image_batch1, label_batch1 = tf.expand_dims(image1, dim=0), tf.expand_dims(label1,
    #                                                                                dim=0)  # Add one batch dimension.
    #
    #     # create net1
    #     net1 = DeepLabResNetModel({'data': image_batch1}, is_training=False,
    #                               num_classes=MAP_NUM_CLASSES)
    #
    #     # Which variables to load.
    #     restore_var1 = tf.global_variables()
    #
    #     # Predictions.
    #     raw_output1 = net1.layers['fc1_voc12']
    #     raw_output_up1 = tf.image.resize_bilinear(raw_output1, tf.shape(image_batch1)[
    #                                                            1:3, ])
    #     raw_output1 = tf.argmax(raw_output_up1, dimension=3)
    #     prob_out1 = tf.nn.softmax(raw_output_up1)
    #     pred1 = tf.expand_dims(raw_output1, dim=3)  # Create 4-d tensor.
    #
    #     # Set up tf session and initialize variables.
    #     config = tf.ConfigProto()
    #     # config.gpu_options.allow_growth = True
    #     sess1 = tf.Session(config=config)
    #     init1 = tf.global_variables_initializer()
    #
    #     sess1.run(init1)
    #     # sess1.run(tf.local_variables_initializer())
    #
    #     # Load weights.
    #     loader = tf.train.Saver(var_list=restore_var1)
    #     if MAP_TRAINED_MODEL is not None:
    #         load(loader, sess1, MAP_TRAINED_MODEL)

    # Start queue threads.
    # threads1 = tf.train.start_queue_runners(coord=coord1, sess=sess1)
    # threads2 = tf.train.start_queue_runners(coord=coord2, sess=sess2)
    #
    # step = 0
    # frames_path = video_path + "Frames/"
    # for file in os.listdir(frames_path):
    for step in range(0, len(new_images)):

        file = frames_path + new_images[step]
        print(file)

        # ------------------------------------------------- #
        # ------------ MAPILLARY SEGMENTATION ------------- #
        # ------------------------------------------------- #

        per = 1
        if not step == 0:
            for l in range(0, per):
                preds, jpg_path, output2_1 = sess1.run(
                    [pred1, title1, prob_out1])  # preds = sess1.run(pred1)
        else:
            preds, jpg_path, output2_1 = sess1.run([pred1, title1,
                                                    prob_out1])  # preds = sess1.run(pred1)     # TODO: Crushes here  OOM when allocating tensor with shape[144,2048,6,9]

        # print(jpg_path)
        preds = np.resize(preds, (1, height, width, 1))

        # compute average probability / class  <================================== TODO better?
        pixels_class = np.zeros(
            (1, MAP_NUM_CLASSES))  # save the number of pixels belong to each class
        sum_prob_class = np.zeros(
            (1, MAP_NUM_CLASSES))  # save the sum of probabilities for each class
        avg_prob_class = np.zeros(
            (1,
             MAP_NUM_CLASSES))  # save average probability for each class (normalize it in the end 0-1)
        pixel_counter = 0
        sh = output2_1.shape
        for i in range(0, sh[1]):
            for j in range(0, sh[2]):
                out = output2_1[0][i][j]
                # print((output2[0][i][j]).shape)
                label_selected = np.argmax(out)
                max_probability = np.max(out)
                # print(label_selected)
                # print("pixel No{}".format(pixel_counter)
                pixels_class[0, label_selected] += 1
                sum_prob_class[0, label_selected] += max_probability
                pixel_counter += 1

        # Calculate average probability values for each label
        for c in range(0, MAP_NUM_CLASSES):
            if pixels_class[0, c] != 0:
                avg_prob_class[0, c] = sum_prob_class[0, c] / pixels_class[0, c]

        # Find which labels are present in current image
        MAP_tags = np.unique(preds)
        print(("tags: {}".format(MAP_tags)))  # TODO: save name of objects localized

        # create dictionary to keep existing tags extracted from segmentation
        existing_tags_dict2 = {}
        existing_tags_dict2["tags"] = []
        existing_tags_list = []
        existing_tags_list_kb = []
        objects = []
        frame_info = []
        frame_objects = []
        temp_tag = ''
        objects = ''
        # print("This image/frame contains:")
        for t in MAP_tags:
            cur_tag_dict = {}
            if not t == 0:
                print(("{} with probability {:.2f}%".format(MAP_labels.get(t),
                                                            100 * avg_prob_class[0, t])))
                cur_tag_dict["type"] = MAP_labels.get(t)
                cur_tag_dict["probability"] = avg_prob_class[0, t]
                bbox_list = []  # TODO: extract bboxes???
                cur_tag_dict["bbox_list"] = bbox_list
                existing_tags_list.append(cur_tag_dict)
                # objects.append(MAP_labels.get(t))
                # frame_info = frame_info + ", " + cur_tag_dict["type"]
                temp_tag = cur_tag_dict["type"]

                if step == 0:
                    cur_tag_dict_kb = {}
                    cur_tag_dict_kb["frame"] = ""
                    cur_tag_dict_kb["type"] = MAP_labels.get(t)
                    cur_tag_dict_kb["probability"] = avg_prob_class[0, t]
                    # bbox_list = []  # TODO: extract bboxes????
                    # cur_tag_dict["bbox_list"] = bbox_list
                    existing_tags_list_kb.append(cur_tag_dict_kb)
                    # objects.append(MAP_labels.get(t))
                    # frame_info = frame_info + ", " + cur_tag_dict["type"]
                    temp_tag = cur_tag_dict["type"]

                frame_objects.append(temp_tag)
                objects = objects + temp_tag + ', '

        frame_info.append([file, objects])

        msk = decode_labels(preds, num_classes=MAP_NUM_CLASSES)
        mask1 = Image.fromarray(msk[0])

        # save segmented image
        if not os.path.exists(masks_path + "MAP_masks/"):
            os.makedirs(masks_path + "MAP_masks/")
        frame_num = os.path.splitext(new_images[step])[0]
        mask1.save(masks_path + "MAP_masks/" + str(frame_num) + '_mask.png')

        # Save mask to server
        file = WebFunctions.encode_file(masks_path + "MAP_masks/" + str(frame_num) + '_mask.png')
        upload_url = WebFunctions.send_post(WebFunctions.masks_url, file, False)

        print("[INFO] Mask is saved in ", upload_url)  # ToDO: Save this url in output JSON

        # Save processed image to server
        file = WebFunctions.encode_file(frames_path + str(
            frame_num) + '.jpg')  # ToDo: It now saves the original frames - Change it to save the processed ones for 3D reconstruction
        upload_url = WebFunctions.send_post(WebFunctions.procImages_url, file, False)

        print("[INFO] Processed frame is saved in ", upload_url)  # ToDO: Save this url in output JSON

        # write frame info in csv   #Todo <-----------------------
        # print(step)
        # start = time.time()
        # if not os.path.exists(masks_path):
        #     os.makedirs(masks_path)
        # with open(masks_path + simmoid + ".csv", 'a', newline='') as myfile:
        #
        #     if step == 0:
        #         writer = csv.writer(myfile)
        #         writer.writerow(["Frame", "Objects"])
        #
        #     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        #     wr.writerow(frame_info[0])

        # step = step + 1
        # end = time.time() - start

        # print("Save info in csv time = {} sec".format(end))

        # # write json file
        # with open(video_dir+"masks/" + input_name + '.json', 'w') as outfile:
        #     json.dump(info, outfile)

        print("[INFO] json type file with frame info is saved!")  # ToDo 2








    return mask, objects, obj_probs

# ------------------------------------------------------------ #
# ------------- FUNCTIONS AND CLASSES NEEDED ----------------- #
# ------------------------------------------------------------ #

# ---------------- for deeplab ---------------- #
def load(saver, sess, ckpt_path):
    '''Load trained weights.

    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    '''
    saver.restore(sess, ckpt_path)
    print(("Restored model parameters from {}".format(ckpt_path)))

def perform_segmentation(sess, img, trained_model, num_classes):
    # Create network.
    net = DeepLabResNetModel({'data': tf.expand_dims(img, dim=0)}, is_training=False, num_classes=num_classes)

    # Which variables to load.
    restore_var = tf.global_variables()

    # Predictions.
    raw_output = net.layers['fc1_voc12']
    raw_output_up1 = tf.image.resize_bilinear(raw_output, tf.shape(img)[0:2, ])
    raw_output_up = tf.argmax(raw_output_up1, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)

    # Load weights.
    loader = tf.train.Saver(var_list=restore_var)
    load(loader, sess, trained_model)

    return raw_output_up1, pred

def convert_from_color_segmentation(arr_3d, palette):
    'converts color masks to grayscale'
    arr_3d = np.array(arr_3d)

    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d

####  Keep CMP_mask as main mask and add the labels except "building" (16) from mapillary mask  ####
def extract_final_mask(mask1, mask2):
    mask1_gray = convert_from_color_segmentation(mask1, MAP_palette)
    mask1_gray = Image.fromarray(mask1_gray)
    # mask1_gray.save("D:/tpistola/V4Design/Codes/Dora_Services/STBOL_new/videos_0ad15d1f-aa75-457b-b935-b7e1bf791b86_masks_MAP/test1.png")

    mask2_gray = convert_from_color_segmentation(mask2, FS_palette)
    mask2_gray = Image.fromarray(mask2_gray)
    # mask2_gray.save("D:/tpistola/V4Design/Codes/Dora_Services/STBOL_new/videos_0ad15d1f-aa75-457b-b935-b7e1bf791b86_masks_FS/test2.png")

    mask1_gray = np.array(mask1_gray)
    mask2_gray = np.array(mask2_gray)
    h, w = mask1_gray.shape

    final_mask = np.zeros((h, w))

    # change labels of FS mask to final (extend MAP labels)
    for i in range(0, h):
        for j in range(0, w):
            if mask2_gray[i, j] == 2:
                mask2_gray[i, j] = 19  # window
            elif mask2_gray[i, j] == 3:
                mask2_gray[i, j] = 20  # door
            elif mask2_gray[i, j] == 1:
                mask2_gray[i, j] = 18  # facade

            # put MAP mask as the base of the final mask
            final_mask[i, j] = mask1_gray[i, j]

            # put windows and doors on the MAP mask
            if mask2_gray[i, j] == 19 or mask2_gray[i, j] == 20:
                final_mask[i, j] = mask2_gray[i, j]

    final_mask = Image.fromarray(final_mask)
    final_mask = final_mask.convert("L")  # needs this because it is grayscale

    return final_mask


# load model for SR   # ToDo: way not load model everytime?
# model1 = load_model('Trained_models/SR/my_newmodel_places_kle_sept.h5')
# print("[INFO] SR model is successfully loaded!")