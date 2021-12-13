from flask import Flask, request, jsonify
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash
from pymongo import MongoClient
import requests
from bson.json_util import ObjectId
import json


# Convert to JSON
class MyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        return super(MyEncoder, self).default(obj)


# API configuration
app = Flask(__name__)
app.json_encoder = MyEncoder
auth = HTTPBasicAuth()  # Basic authentication
MONGO_URI = "" #need credentials (contact author)

# Connect to MongoDB
db = MongoClient(MONGO_URI) #, ssl_cert_reqs=ssl.CERT_NONE)
print("[INFO] Successfully connected to MongoDB!")

# Define usernames and passwords for access in the api
# users = {
    
# }

# The function below receives the username and password sent by the client.
# If the credentials belong to a user, then the function should return the user object.
@auth.verify_password
def verify_password(username, password):
    if username in users and \
            check_password_hash(users.get(username), password):
        return username


####################### POST REQUESTS ###############################
#####################################################################


@app.route('/request', methods=['POST'])  # Visual Analysis Request
@auth.login_required
def post_request():

    # Get input request
    input = request.json
    print("Input JSON received: ", input)

    input_request = {}
    input_request["input"] = input
    input_request["status"] = 0  # 0: Pending

    # Check if the input request exists
    result = db.ImagesDB.InputQueue.find_one(filter={"input.simmoid": input["simmoid"]})
    # result = db.XR4D_Visual_Analysis_DB.InputQueue.find_one(filter={"input.simmoid": input["simmoid"]})
    print("result: ", result)
    if result is None:
        # Add new request in MongoDB
        db.ImagesDB.InputQueue.insert_one(input_request)
        # db.XR4D_Visual_Analysis_DB.InputQueue.insert_one(input_request)
        # db.ImagesDB.InputQueue.update(input_request, input_request, upsert=True)   # This line inserts one without having duplicates.
    else:
        return "Request for simmoid {} is already in queue with status {}.".format(input["simmoid"], result["status"]), 200
        # TODO: send output that is saved in MongoDB

    print("Request added in queue.")
    return json.dumps({"Request saved": True}), 200


############################ GET ####################################
#####################################################################
@app.route('/output/<simmoid>', methods=['GET'])  # Visual Analysis Output Retrieval
@auth.login_required
# Retrieve the analysis metadata by simmoid. Saved in visual analysis MongoDB (and in KB).
def get_output(simmoid):

    # retrieved_output = db.ImagesDB.VisualsOutputKB.find_one(filter={"header.simmoid": simmoid})
    retrieved_output = db.XR4D_Visual_Analysis_DB.VisualsOutputKB.find_one(filter={"header.simmoid": simmoid})

    output = {}
    output["header"] = retrieved_output["header"]
    output["shotInfo"] = retrieved_output["shotInfo"]

    return json.dumps(output), 200


@app.route('/multimedia/processed_frames/<simmoid>', methods=['GET'])  # Visual Analysis Output (sent to 3D reconstruction) Retrieval
@auth.login_required
# Retrieve the analysis metadata by simmoid. Saved in visual analysis MongoDB (and sent to 3D Rec).
def get_proc_frames(simmoid):

    # retrieved_output = db.ImagesDB.VisualsOutput3D.find_one(filter={"header.simmoid": simmoid})
    retrieved_output = db.XR4D_Visual_Analysis_DB.VisualsOutput3D.find_one(filter={"header.simmoid": simmoid})

    output = {}
    output["header"] = retrieved_output["header"]
    output["shotInfo"] = retrieved_output["shotInfo"]

    return json.dumps(output), 200


@app.route('/multimedia/masks/<simmoid>', methods=['GET'])  # Visual Analysis Output (sent to 3D reconstruction) Retrieval
@auth.login_required
# Retrieve the analysis metadata by simmoid. Saved in visual analysis MongoDB (and sent to 3D Rec).
def get_masks(simmoid):

    # retrieved_output = db.ImagesDB.Masks.find_one(filter={"header.simmoid": simmoid})
    retrieved_output = db.XR4D_Visual_Analysis_DB.Masks.find_one(filter={"header.simmoid": simmoid})

    output = {}
    output["header"] = retrieved_output["header"]
    output["shotInfo"] = retrieved_output["shotInfo"]

    return json.dumps(output), 200


@app.route('/status', methods=['GET'])  # Visual Analysis Status
@auth.login_required
# Retrieve the status of the Visual Analysis service.
def get_status():

    # {
    #     "status": enumerate(
    #         "operational",
    #         "degraded",
    #         "failed"
    #     ),
    #     "status_description": string
    # }
    status = "Operational"

    return json.dumps(status), 200


@app.route('/status/log', methods=['GET'])  # Visual Analysis Status/Log
@auth.login_required
# Retrieve the status/log of the Visual Analysis service.
def get_status_log():

    with open("full_log_file.json", "r") as file_read:
        past_log_data = json.load(file_read)

    return json.dumps(past_log_data)


if __name__ == '__main__':
    app.run(host="160.40.53.24", port=6005)


