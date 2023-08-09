import base64, io
from PIL import Image

from tiled.client import from_uri
from tiled.client.node import Node

from file_manager.dataset.dataset import Dataset


class TiledDataset(Dataset):
    def __init__(self, uri, type='tiled', tags=[], project=None, **kwargs):
        '''
        Definition of a tiled data set
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
        rawBytes = io.BytesIO()
        tiled_data = from_uri(self.uri)
        img = tiled_data.export(rawBytes, format='jpeg')
        rawBytes.seek(0)        # return to the start of the file
        if export=='pillow':
            return Image.open(rawBytes), self.uri
        img = base64.b64encode(rawBytes.read())
        return f'data:image/jpeg;base64,{img.decode("utf-8")}', self.uri

    @staticmethod
    def browse_data(tiled_uri, browse_format, tiled_uris=[], tiled_client=None):
        '''
        Retrieve a list of nodes from tiled URI
        Args:
            tiled_uri:          Tiled URI from which data should be retrieved
            browse_format:      List of file formats/extensions of interest, defaults to FORMATS
            tiled_uris:         List of current tiled URIs, which is used when the method is run
                                recursively, defaults to []
            tiled_client:       Current tiled client, which is used when the method is run
                                recursively, defaults to None
        Returns:
            tiled_uris:         List of tiled URIs found in tiled client
        '''
        if not tiled_client:
            tiled_client = from_uri(tiled_uri)
        if isinstance(tiled_client, Node):
            nodes = list(tiled_client)
            for node in nodes:
                mod_tiled_uri = TiledDataset.update_tiled_uri(tiled_uri, node)
                if browse_format != '**/' and isinstance(tiled_client[node], Node):
                    # Recursively browse data if a node is encounteres
                    TiledDataset.browse_data(mod_tiled_uri, browse_format, tiled_uris,
                                            tiled_client=tiled_client[node])
                else:
                    tiled_uris.append(mod_tiled_uri)
        else:
            tiled_uris.append(tiled_uri)
        return tiled_uris
    
    @staticmethod
    def update_tiled_uri(tiled_uri, node):
        '''
        Update tiled URI to reflect nodes
        '''
        if '/api/v1/metadata' in tiled_uri:
            return f'{tiled_uri}/{node}'
        elif tiled_uri[-1] == '/':
            return f'{tiled_uri}api/v1/metadata/{node}'
        else:
            return f'{tiled_uri}/api/v1/metadata/{node}'
