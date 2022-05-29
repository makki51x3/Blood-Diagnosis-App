from app import app
import unittest
from flask import jsonify
import base64

class FlaskTestCase(unittest.TestCase):
    # Ensure that flask was set up correctly
    def test_index(self):
        app.testing = True
        tester = app.test_client()
        with open(r"C:\Users\1999m\Desktop\lira competition\LIRA\Blood-Diagnosis-App\API\MyPythonScripts\testImage.png", "rb") as img_file:
            base64Img = base64.b64encode(img_file.read())
        base64Img = "data:image/png;base64,"+str(base64Img.decode('utf-8'))

        response = tester.post(
            "http://127.0.0.1:5000"+"/send-image/", 
            json={"selectedImg":"{}".format(base64Img)},
        )

        # check that the request succeeded
        self.assertEqual(response.status_code, 200)

        # check that mask, label, and feature exist in the response
        self.assertIn(b'info', response.data, 'response is:\n{}'.format(response.data))
        self.assertIn(b'"maskBase64":"data:image/png;base64,', response.data, 'response is:\n{}'.format(response.data))
        self.assertIn(b'"labelBase64":"data:image/png;base64,', response.data, 'response is:\n{}'.format(response.data))

        # check that all features exist in info
        self.assertIn(b'area', response.data, 'response is:\n{}'.format(response.data))
        self.assertIn(b'equivalent_diameter', response.data, 'response is:\n{}'.format(response.data))
        self.assertIn(b'perimeter', response.data, 'response is:\n{}'.format(response.data))
        self.assertIn(b'eccentricity', response.data, 'response is:\n{}'.format(response.data))
        self.assertIn(b'moments_hu-0', response.data, 'response is:\n{}'.format(response.data))
        self.assertIn(b'moments_hu-1', response.data, 'response is:\n{}'.format(response.data))
        self.assertIn(b'moments_hu-2', response.data, 'response is:\n{}'.format(response.data))
        self.assertIn(b'moments_hu-3', response.data, 'response is:\n{}'.format(response.data))
        self.assertIn(b'moments_hu-4', response.data, 'response is:\n{}'.format(response.data))
        self.assertIn(b'moments_hu-5', response.data, 'response is:\n{}'.format(response.data))
        self.assertIn(b'moments_hu-6', response.data, 'response is:\n{}'.format(response.data))
        self.assertIn(b'orientation', response.data, 'response is:\n{}'.format(response.data))

if __name__ == '__main__':
    unittest.main()