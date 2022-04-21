from app import app
import unittest
from flask import jsonify

class FlaskTestCase(unittest.TestCase):

    # Ensure that flask was set up correctly
    def test_index(self):
        app.testing = True
        tester = app.test_client()
        response = tester.post("http://127.0.0.1:5000"+"/send-image/", data=jsonify({"selectedImg":"https://findicons.com/files/icons/1637/file_icons_vs_2/256/url.png"}))
        self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()