from datetime import datetime
import pathlib
from typing import List
from PIL import Image
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

    def init_from_dict(self, data: List, api_key=None):
        '''
        Initialize the object from a dictionary
        '''
        self.data = []
        for item in data:
            if item['type']=='tiled':
                if 'api_key' not in item:
                    item['api_key'] = api_key
                self.data.append(TiledDataset(**item))
            else:
                self.data.append(LocalDataset(**item))
    
    def init_from_splash(self, splash_uri, project_id=None, api_key=None):
        '''
        Initialize the object from splash
        '''
        datasets = requests.post(splash_uri,
                                 json={'project': project_id},
                                 params={'page[limit]': 100000}).json()
        self.init_from_dict(datasets, api_key)
        
    @staticmethod
    def browse_data(data_type, browse_format, dir_path=None, tiled_uri=None, api_key=None):
        '''
        Browse data according to browse format and data type
        Args:
            data_type:          Tiled or local
            browse_format:      File format to retrieve during this process
            dir_path:           Directory path if data_type is local
            tiled_uri:          Tiled URI if data_type is tiled
            api_key:            Tiled API key
        Returns:
            data:               Retrieve Dataset according to data_type and browse format
        '''
        if data_type == 'tiled':
            uris = TiledDataset.browse_data(tiled_uri, browse_format, api_key=api_key, tiled_uris=[])
            data = [TiledDataset(uri=item, api_key=api_key) for item in uris]
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
            
    def add_to_splash(self, splash_uri, project_id=str(uuid4())):
        '''
        POST list of data sets to splash-ml with a corresponding project_id
        Args:
            splash_uri:         URI to splash-ml service
            project_id:         Data project id
        '''
        validate_project_id = False
        data_project_uris = [dataset.uri for dataset in self.data]
        # Get the project ID of first element
        splash_datasets = requests.post(
            f'{splash_uri}/datasets/search', json={'uris': [data_project_uris[0]]}).json()
        for splash_dataset in splash_datasets:
            # Check that all the data sets in this project match the id
            splash_project = requests.post(
                f'{splash_uri}/datasets/search?page%5Blimit%5D={len(self.data)+1}', 
                json={'uris': data_project_uris,
                      'project': splash_dataset['project']}
                ).json()
            if len(splash_project) == len(self.data):
                project_id = splash_dataset['project']
                validate_project_id = True
                break
        self.project = project_id
        datasets_dict = []
        for dataset in self.data:
            dataset.project = self.project
            datasets_dict.append(dataset.__dict__)
        if not validate_project_id:
            splash_uids = requests.post(f'{splash_uri}/datasets', json=datasets_dict).json()
            # update data sets uids to match splash-ml
            self.data = [{**d, 'uid': uid} for d, uid in zip(datasets_dict, splash_uids)]
        pass
    
    def tiled_to_local_project(self, data_path, project_id):
        '''
        Convert a tiled data project to a local project while saving each dataset to filesystem
        Args:
            data_path:      Target data path for new local data project
            project_id:     Project ID for new local data project
        Returns:
            local_data_project
        '''
        local_datasets = []
        for dataset in self.data:
            img, _ , _ = dataset.read_data(export='pillow')
            new_uri = f'{data_path}/{dataset.uid}.tif'
            img.save(new_uri)                   # Save data to new path
            local_datasets.append(LocalDataset(new_uri, project=project_id))
        local_data_project = DataProject(data=local_datasets,
                                         project_id=project_id)
        return local_data_project
