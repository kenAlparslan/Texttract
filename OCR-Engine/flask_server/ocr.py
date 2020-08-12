# -*- coding: utf-8 -*-

import pytesseract
from localfile import session

from PIL import Image
from PIL import ImageFilter
from io import BytesIO

def process_image(url):
    '''
    Returns the base-64 string of image given an url

        Parameters:
                url (string): url for the image

        Returns:
                (str): base64 string for the image
    '''
    image = _get_image(url)
    image.filter(ImageFilter.SHARPEN)
    return pytesseract.image_to_string(image)


def _get_image(url, session=session):
    '''
    Opens and identifies the given image file with given an url

        Parameters:
                url (string): url for the image
                session(Flask object): Flask session, optional

        Returns:
                (Image object): Pillow image object
    '''
    return Image.open(BytesIO(session.get(url).content))