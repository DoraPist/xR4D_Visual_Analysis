import os
import pymongo
from bson.json_util import dumps
import time
import json
from visual_analysis_functions_3drec_new import visual_analysis

# Connect to MongoDB
MONGO_URI = "mongodb+srv://dora_user:wk9NmSQIPqgXNf8O@doracluster.se2tb.mongodb.net/ImagesDB?ssl=true&ssl_cert_reqs=CERT_NONE"  # Atlas MongoDB
# MONGO_URI = "mongodb://xr4d_visuals_user:B6^M2qAe@xr4drama.iti.gr:27017/XR4D_Visual_Analysis_DB?replicaSet=rs"  # Local MongoDB
client = pymongo.MongoClient(MONGO_URI)

print("[INFO] ******* Visual analysis listener is ready!")

# Listen to changes in InputQueue collection
change_stream = client.ImagesDB.InputQueue.watch()
# change_stream = client.XR4D_Visual_Analysis_DB.InputQueue.watch()
for change in change_stream:

    request = dumps(change)
    json_request = json.loads(request)
    # print(json_request)
    # print('') # for readability only
    print("\n\n [INFO] ******* Visual analysis listener received a new request: \n")

    print(json_request["fullDocument"]["input"])
    simmoid = json_request["fullDocument"]["input"]["simmoid"]

    # print("\n[INFO] Analyzing input. Please wait...\n")
    # Call the visual analysis function
    visual_analysis(simmoid, sendtoKB=False, sendtoRec=False)

    print("\n[INFO] ******* Visual analysis listener waits for request...")
    # time.sleep(15)