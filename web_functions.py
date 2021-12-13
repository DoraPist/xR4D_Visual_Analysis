import requests
import json
from requests.auth import HTTPBasicAuth
import base64
from pytube import YouTube
import os

class WebFunctions:

    # Credentials for http://xr4drama.iti.gr:5002/fileUpload/{}
    USERNAME = ""    # Need credentials (contact author)
    API_PASSWORD = ""
    DOWLOAD_PASSWORD = ""

    masks_url = "http://xr4drama.iti.gr:5002/fileUpload/masks"
    procImages_url = "http://xr4drama.iti.gr:5002/fileUpload/processed_images"

    @staticmethod
    def get_json(simmo_id, entity):
        url = "http://xr4drama.iti.gr:5003/simmo/" + entity + "/" + simmo_id
        json_file = requests.get(url, auth=HTTPBasicAuth(WebFunctions.USERNAME, WebFunctions.API_PASSWORD))
        return json_file


    @staticmethod
    def send_post(url, file, is_json=True):
        """
        Method to serve post requests for the xR4DRAMA project
        :param url: the url to send the request
        :param file: the file content to upload
        :param is_json: set that to false for posting plain text or to upload base-64 encoded files
        :return:
        """

        # send request and return response
        # if is_json:
        #     # if body is string, convert it to dictionary
        #     if not type(body) is dict:
        #         try:
        #             body = json.loads(body)
        #         except ValueError as e:
        #             print("Body parameter is not in JSON format!")
        #             return '{}'
        #     r = requests.post(url, files=body, auth=HTTPBasicAuth(USERNAME, PASSWORD))
        # else:
        #     r = requests.post(url, files=body,  headers={'Content-Type': 'text/plain'}, auth=HTTPBasicAuth(USERNAME, PASSWORD))
        r = requests.post(url, files=file, auth=HTTPBasicAuth(WebFunctions.USERNAME, WebFunctions.DOWLOAD_PASSWORD))
        if r.status_code == 200:
            return r.text
        else:
            print("Error!")
            print("Response code:", r.status_code)
            print("Message:", r.text)
            return '{}'

    @staticmethod
    def image_get(url, save_path):
        """
        Method to get image from filestorage for the xR4DRAMA project
        :param url: the url to get the image
        :return:
        """
        r = requests.get(url, auth=HTTPBasicAuth(WebFunctions.USERNAME, WebFunctions.DOWLOAD_PASSWORD))
        if not r.status_code == 200:
            print("Error!")
            print("Response code:", r.status_code)
            print("Message:", r.text)
            # return '{}'
        image_result = open(save_path, 'wb')
        image_result.write(r.content)
        print("Image was saved in path: ", save_path)

    @staticmethod
    def encode_file(filename):
        """
        Method to encode image using base64 to send post request
        :param filename: filename of the image
        :return: encoded image file
        """
        file = {'file': open(filename, 'rb')}
        return file

    @staticmethod
    def download_from_url(url, path, filename):
        """
        Method to serve file downloads for the V4Design project
        :param url: the url of the file to download
        :param filename: the destination (path) of the downloaded file
        :return:
        """

        if url.find('youtube.com') != -1:
            # Download video from YouTube
            yt = YouTube('https://www.youtube.com/watch?v=7BjeppD53J4')
            yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution')\
                .desc().first().download(output_path=path, filename=filename)
            print("[INFO] File has been successfully saved in ", path + filename)
            return True
        else:
            r = requests.get(url, allow_redirects=True,
                             auth=HTTPBasicAuth(WebFunctions.USERNAME, WebFunctions.DOWLOAD_PASSWORD))
            if r.status_code == 200:
                open(path+filename, 'wb').write(r.content)
                print("[INFO] File has been successfully saved in ", path+filename)
                return True
            else:
                print("Error downloading file!")
                print("Response code:", r.status_code)
                return False
