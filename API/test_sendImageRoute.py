from app import app
import unittest
from flask import jsonify
import base64

class FlaskTestCase(unittest.TestCase):
    
    # Ensure that flask was set up correctly
    def test_index(self):
        app.testing = True
        tester = app.test_client()
        with open(r"Blood-Diagnosis-App\API\MyPythonScripts\testImage.png", "rb") as img_file:
            base64Img = base64.b64encode(img_file.read())
        base64Img = "data:image/png;base64,"+str(base64Img.decode('utf-8'))

        response = tester.post(
            "http://127.0.0.1:5000"+"/send-image/", 
            json={"selectedImg":"{}".format(base64Img)},
        )
        print(response)
        self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()