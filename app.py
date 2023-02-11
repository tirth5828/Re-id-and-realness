import cv2
from flask import Flask, request
import numpy as np
import json 
import requests


app = Flask(__name__)



def realness_score(image):
    image = cv2.resize(image, (640, 480))
    jpeg_image = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])[1]
    jpeg_image = cv2.imdecode(jpeg_image, cv2.IMREAD_UNCHANGED)
    difference = cv2.absdiff(image, jpeg_image)
    difference = cv2.normalize(difference, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    avg_pixel_value = np.mean(difference)
    return 1 - (avg_pixel_value / 255)

def calculate_reid_score(faces1 , faces2 , img1 , img2):
        try:
            (x1, y1, w1, h1) = faces1
            (x2, y2, w2, h2) = faces2
        except:
            return "Error: Could not get coordinates of face"

        # Crop the faces from the images
        try:
            face1 = img1[y1:y1 + h1, x1:x1 + w1]
            face2 = img2[y2:y2 + h2, x2:x2 + w2]
        except:
            return "Error: Could not crop face from image"

        
        # Convert the faces to grayscale
        try:
            face1_gray = cv2.cvtColor(face1, cv2.COLOR_BGR2GRAY)
            face2_gray = cv2.cvtColor(face2, cv2.COLOR_BGR2GRAY)
        except:
            return "Error: Could not convert face to grayscale"

        
        # Use the ORB feature detector to find keypoints and descriptors in both faces
        try:
            orb = cv2.ORB_create()
        except:
            return "Error: Could not create ORB object"
        
        try:
            kp1, des1 = orb.detectAndCompute(face1_gray, None)
            kp2, des2 = orb.detectAndCompute(face2_gray, None)
            
            print("len(kp1) == len(kp2)" , len(kp1) == len(kp2))
            print("len(kp1)" , len(kp1))
            print("len(kp2)" , len(kp2))
        except:
            return "Error: Could not detect and compute keypoints and descriptors"

        # Use the BFMatcher to match the keypoints and descriptors between the two faces
        try:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        except:
            return "Error: Could not create BFMatcher object"


        try:
            matches = bf.match(des1, des2)
        except Exception as e:
            print(e)
            return "Error: Could not match keypoints and descriptors"
        
        # Sort the matches by distance and keep only the top N matches
        try:
            matches = sorted(matches, key=lambda x: x.distance)
        except:
            return "Error: Could not sort matches"
        
        # try:
        #     N = 20 # Number of matches to keep
        #     matches = matches[:N]
        # except:
        #     return "Error: Could not keep top N matches"

        
        # Calculate the re-identification score as the ratio of matched keypoints to total keypoints
        try:
            score = (len(matches) * 2) / (len(kp1) + len(kp2))
        except:
            print("len(matches) , len(kp1)",len(matches) , len(kp1) )
            return "Error: Could not calculate re-identification score"
        
        try:
            return score
        except:
            return "Error: Could not add similarity score to output"



@app.route("/predict", methods=["POST"])
def predict():
    
    output = {}
    # Get the image from the request
    try:
        # image1 = request.files.get("image1").read()
        # image2 = request.files.get("image2").read()
        # print(type(image1))
        # print("image1 == image2" , image1 == image2)
        
        
        
        images = request.get_json()
        # return str(images)

        response1 = requests.get(images['image1'])
        image1 = response1.content

        response2 = requests.get(images['image2'])
        image2 = response2.content
    except:
        return "Error: No image found in request"
    
    
    # Convert the image to a numpy array
    try:
        image_array1 = np.frombuffer(image1, np.uint8)
        image_array2 = np.frombuffer(image2, np.uint8)
    except:
        return "Error: Could not convert image to numpy array"

    try:    
        realness_score1 = realness_score(image_array1)
        realness_score2 = realness_score(image_array2)
    except:
        return "Error: Could not calculate realness score"
    
    try:
        output["realness_score1"] = realness_score1
        output["realness_score2"] = realness_score2
    except:
        return "Error: Could not add realness score to output"
    
    try:
        img1 = cv2.imdecode(image_array1, cv2.IMREAD_COLOR)
        img2 = cv2.imdecode(image_array2, cv2.IMREAD_COLOR)
    except:
        return "Error: Could not decode image"
    
    
    try:
        if img1.shape[0] == 0 or img1.shape[1] == 0:
            return "Error: Input image is empty"
        
        if img2.shape[0] == 0 or img2.shape[1] == 0:
            return "Error: Input image is empty"
    except:
        return "Error: Could not get shape of image"


    # Load the pre-trained Haar Cascade classifier for face detection
    try:
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface.xml")
    except:
        return "Error: Could not load Haar Cascade classifier"


    # Detect faces in both images
    try:
        faces1 = face_cascade.detectMultiScale(img1)
        faces2 = face_cascade.detectMultiScale(img2)
        
        print(faces1)
    except:
        return "Error: Could not detect faces in image"
    
    print("len(faces1)" , len(faces1))
    print("len(faces2)" , len(faces2))

    # Check if at least one face was detected in both images
    if len(faces1) > 0 and len(faces2) > 0:
        score = []
        error = []
        for f1 in faces1:
            for f2 in faces2:
                result = calculate_reid_score(f1,f2,img1,img2)
                print(type(result))
                if isinstance(result, str):
                    error.append(result)
                else:
                    score.append(result)
        
        try:
            output["similarity_score"] = score
            output["error"] = error
        except:
            return "Error: Could not add similarity score to output"
        
        try:
            json_object = json.dumps(output, indent = 4) 
        except:
            return "Error: Could not convert output to JSON"
        
        return  json_object

    else:
        return "No faces detected in one or both images"

if __name__ == '__main__':
    app.run()
