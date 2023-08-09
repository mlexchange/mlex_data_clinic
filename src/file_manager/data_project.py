from datetime import datetime
import json
from typing import List

import requests
from uuid import uuid4

from file_manager.dataset.local_dataset import LocalDataset
from file_manager.dataset.tiled_dataset import TiledDataset


class DataProject:
    def __init__(self, data: List = [], project_id = None):
        '''
        Definition of a DataProject
        Args:
            data:       List of data sets within the project
            project_id: Project ID to track the project of interest
        '''
        self.data = data
        self.project_id = project_id
        pass

    def init_from_dict(self, data: List):
        '''
        Initialize the object from a dictionary
        '''
        self.data = []
        for item in data:
            if item['type']=='tiled':
                self.data.append(TiledDataset(**item))
            else:
                self.data.append(LocalDataset(**item))
    
    def init_from_splash(self, splash_uri):
        '''
        Initialize the object from splash
        '''
        datasets = requests.post(splash_uri[0], json={'project': splash_uri[1]}).json()
        self.init_from_dict(datasets)
        
    @staticmethod
    def browse_data(data_type, browse_format, dir_path=None, tiled_uri=None):
        '''
        Browse data according to browse format and data type
        Args:
            data_type:          Tiled or local
            browse_format:      File format to retrieve during this process
            dir_path:           Directory path if data_type is local
            tiled_uri:          Tiled URI if data_type is tiled
        Returns:
            data:               Retrieve Dataset according to data_type and browse format
        '''
        if data_type == 'tiled':
            uris = TiledDataset.browse_data(tiled_uri, browse_format, tiled_uris=[])
            data = [TiledDataset(uri=item) for item in uris]
        else:
            if browse_format=='*':
                uris = LocalDataset.filepaths_from_directory(dir_path)
            else:
                if browse_format == '**/*.jpg':             # Add variations of the file extensions
                    browse_format = ['**/*.jpg', '**/*.jpeg']
                elif browse_format == '**/*.tif':
                    browse_format = ['**/*.tif', '**/*.tiff']
                # Recursively call the method if a subdirectory is encountered
                uris = LocalDataset.filepaths_from_directory(dir_path, browse_format)
            data = [LocalDataset(uri=str(item)) for item in uris]
        return data
    
    def get_dict(self):
        '''
        Retrieve the dictionary from the object
        '''
        data_project_dict = [dataset.__dict__ for dataset in self.data]
        return data_project_dict
    
    def get_table_dict(self):
        '''
        Retrieve a curated dictionary for the dash table without tags due to imcompatibility with 
        dash table and a list of items in a cell
        '''
        data_table_dict = [{"uri": dataset.uri, "type": dataset.type} for dataset in self.data]
        return data_table_dict
    

    def get_event_id(self, splash_uri):
        '''
        Post a tagging event in splash-ml
        Args:
            splash_uri:         URI to splash-ml service
        Returns:
            event_uid:          UID of tagging event
        '''
        event_uid = requests.post(f'{splash_uri}/events',               # Post new tagging event
                                  json={'tagger_id': 'labelmaker',
                                        'run_time': str(datetime.utcnow())}).json()['uid']
        return event_uid
            

    def add_to_splash(self, splash_uri):
        '''
        Add list of data sets to splash-ml
        Args:
            splash_uri:         URI to splash-ml service
        '''
        validate_project_id = True
        data_project_uris = [dataset.uri for dataset in self.data]
        # Get the project ID of first element
        splash_dataset = requests.post(
            f'{splash_uri}/datasets/search', json={'uris': [data_project_uris[0]]}).json()
        project_id = splash_dataset[0]['project']
        # Check that all the data sets in this project match the id
        splash_datasets = requests.post(
            f'{splash_uri}/datasets/search?page%5Blimit%5D={len(self.data)+1}', 
            json={'uris': data_project_uris,
                  'project': project_id}
            ).json()
        if len(splash_datasets) != len(self.data):  # when there is no match, generate new project id
            project_id = str(uuid4())
            validate_project_id = False
        self.project = project_id
        for dataset in self.data:
            dataset.project = self.project
            if not validate_project_id:
                dataset_dict=dataset.__dict__
                requests.post(f'{splash_uri}/datasets', json=dataset_dict)
        pass
    
