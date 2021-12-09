import requests
import json
from requests.auth import HTTPBasicAuth
import base64
from pytube import YouTube
import os

class WebFunctions:

    # Credentials for http://xr4drama.iti.gr:5002/fileUpload/{}
    USERNAME = "xr4d_user"
    API_PASSWORD = "kJ01H_Pr16zVHR@q"
    DOWLOAD_PASSWORD = "Lc74H_Pr13zHR@qV"    #"kJ01H_Pr16zVHR@q"

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



    # @staticmethod
    # def decode_file(file):
    #     """
    #      Method to decode downloaded image using base64 from file storage
    #      :param file: encoded image file
    #      :param save_path: path to save the downloaded image
    #      :return: encoded image file
    #      """
    #     # convert bytes to ndarray
    #     # image = base64.decodebytes(file)
    #     print(file)
    #     # Save image in file.
    #     image_result = open("Downloads/" + "test25.jpg", 'wb')
    #     image_result.write(file)
    #     print("Image was saved in path: ", "Downloads/" + "test25.jpg")

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

# url = "http://xr4drama.iti.gr:5003/simmo/video/cf798436-e82a-4af5-8de0-c6faeb8069cb"
# url = "http://xr4drama.iti.gr:5003/simmo/twitter_post/1408037357379596290"
# json_file = requests.get(url, auth=HTTPBasicAuth(WebFunctions.USERNAME, WebFunctions.API_PASSWORD))
# print(json_file.content)

# url = "http://xr4drama.iti.gr:5002/download/youtube/9678fe0f-7e9e-4ca4-95fb-854aa90b4966.mp4"
# WebFunctions.download_from_url(url, "Downloads/9678fe0f-7e9e-4ca4-95fb-854aa90b4966.mp4")
# video = requests.get(url, auth=HTTPBasicAuth(WebFunctions.USERNAME, WebFunctions.PASSWORD))
# print(video)

# url = "https://video.twimg.com/ext_tw_video/1408037303193309186/pu/vid/720x1280/ja75AyF87fzS0m2S.mp4?tag=12"
# url = "https://www.youtube.com/watch?v=e0ps-8cbY4I"
# # online_video = requests.get(url)
# online_video = requests.get(url, allow_redirects=True,
#                          auth=HTTPBasicAuth(WebFunctions.USERNAME, WebFunctions.DOWLOAD_PASSWORD))
# print(online_video)
#
# if online_video.status_code == 200:
#     open("Downloads/twitter_video2.mp4", 'wb').write(online_video.content)


# # Test
# # # Define an image object with the location.
# filename = "Corfu2/Frames/frame_25.jpg"
# #
# # Save image
# # Open the image in read-only format.
# file = {'file': open(filename, 'rb')}
#
# # request url for mask
# url = "http://xr4drama.iti.gr:5002/fileUpload/masks"
# uploaded_file_url = WebFunctions.send_post(url, file, False)
# # r = requests.post(url, files=file, auth=HTTPBasicAuth(USERNAME, PASSWORD))
#
# print(uploaded_file_url)
# # print(r.content)
#
# r = WebFunctions.image_get(uploaded_file_url)
# print(r)


