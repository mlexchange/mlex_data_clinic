import base64, io
from functools import reduce

import glob
import os 
import pathlib
from PIL import Image

from file_manager.dataset.dataset import Dataset

# List of allowed and not allowed formats
FORMATS = ['**/*.png', '**/*.jpg', '**/*.jpeg', '**/*.tif', '**/*.tiff']
NOT_ALLOWED_FORMATS = ['**/__pycache__/**', '**/.*']


class LocalDataset(Dataset):
    def __init__(self, uri, type='file', tags=[], project=None, **kwargs):
        '''
        Definition of a local data set
        '''
        super().__init__(uri, type, tags, project)
        pass

    def read_data(self, export='base64'):
        '''
        Read data set
        Returns:
            Base64/PIL image
            Dataset URI
        '''
        filename = self.uri
        img = Image.open(filename)
        if export == 'pillow':
            return img, self.uri
        img = img.resize((300, 300))
        rawBytes = io.BytesIO()
        img.save(rawBytes, "JPEG")
        rawBytes.seek(0)        # return to the start of the file
        img = base64.b64encode(rawBytes.read())
        file_ext = filename[filename.find('.')+1:]
        return 'data:image/'+file_ext+';base64,'+img.decode("utf-8"), self.uri
    
    @staticmethod
    def filepaths_from_directory(directory, formats=FORMATS, sort=True):
        '''
        Retrieve a list of filepaths from a given directory
        Args:
            directory:      Directory from which datapaths will be retrieved according to formats
            formats:        List of file formats/extensions of interest, defaults to FORMATS
            sort:           Sort output list of filepaths, defaults to True
        Returns:
            paths:          List of filepaths in directory
        '''
        if type(formats) == str:    # If a single format was selected, adapt to list
            formats = [formats]
        # Find paths that match the format of interest
        paths = list(reduce(lambda list1, list2: list1 + list2, \
                            (glob.glob(str(directory)+'/'+t, recursive=True) for t in formats)))
        # Find paths that match the not allowed file/directory formats
        not_allowed_paths = list(reduce(lambda list1, list2: list1 + list2, \
                            (glob.glob(str(directory)+'/'+t, recursive=True) for t in NOT_ALLOWED_FORMATS)))
        # Remove not allowed filepaths from filepaths of interest
        paths = list(set(paths) - set(not_allowed_paths))
        if sort:
            paths.sort()
        return paths
