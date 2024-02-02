import torch
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_array
import numpy as np
import imp
import os
from bob.extension.download import get_file

class PyTorchModel(TransformerMixin, BaseEstimator):
    """
    Base Transformer using pytorch models


    Parameters
    ----------

    checkpoint_path: str
       Path containing the checkpoint

    config:
        Path containing some configuration file (e.g. .json, .prototxt)

    preprocessor:
        A function that will transform the data right before forward. The default transformation is `X/255`

    """

    def __init__(
        self,
        checkpoint_path=None,
        config=None,
        preprocessor=lambda x: (x - 127.5) / 128.0,
        device='cpu',
        **kwargs
    ):

        super().__init__(**kwargs)
        self.checkpoint_path = checkpoint_path
        self.config = config
        self.model = None
        self.preprocessor_ = preprocessor
        self.device=device

    def preprocessor(self, X):
        X = self.preprocessor_(X)
        return X
        
    def transform(self, X):
        """__call__(image) -> feature

        Extracts the features from the given image.

        **Parameters:**

        image : 2D :py:class:`numpy.ndarray` (floats)
        The image to extract the features from.

        **Returns:**

        feature : 2D or 3D :py:class:`numpy.ndarray` (floats)
        The list of features extracted from the image.
        """
        if self.model is None:
            self._load_model()
            
            self.model.eval()
            
            self.model.to(self.device)
            for param in self.model.parameters():
                param.requires_grad=False
                
        # X = check_array(X, allow_nd=True)
        # X = torch.Tensor(X)
        X = self.preprocessor(X)

        return self.model(X)#.detach().numpy()


    def __getstate__(self):
        # Handling unpicklable objects

        d = self.__dict__.copy()
        d["model"] = None
        return d

    def _more_tags(self):
        return {"stateless": True, "requires_fit": False}
    
    def to(self,device):
        self.device=device
        
        if self.model !=None:            
            self.model.to(self.device)



def _get_iresnet_file():
    urls = [
        "https://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/pytorch/iresnet-91a5de61.tar.gz",
        "http://www.idiap.ch/software/bob/data/bob/bob.bio.face/master/pytorch/iresnet-91a5de61.tar.gz",
    ]

    return get_file(
        "iresnet-91a5de61.tar.gz",
        urls,
        cache_subdir="data/pytorch/iresnet-91a5de61/",
        file_hash="3976c0a539811d888ef5b6217e5de425",
        extract=True,
    )

class IResnet100(PyTorchModel):
    """
    ArcFace model (RESNET 100) from Insightface ported to pytorch
    """

    def __init__(self,  
                preprocessor=lambda x: (x - 127.5) / 128.0, 
                device='cpu'
                ):

        self.device = device
        filename = _get_iresnet_file()

        path = os.path.dirname(filename)
        config = os.path.join(path, "iresnet.py")
        checkpoint_path = os.path.join(path, "iresnet100-73e07ba7.pth")

        super(IResnet100, self).__init__(
            checkpoint_path, config, device=device
        )

    def _load_model(self):

        model = imp.load_source("module", self.config).iresnet100(self.checkpoint_path)
        self.model = model
        

def get_FaceRecognition_transformer(device='cpu'):
    FaceRecognition_transformer = IResnet100(device=device)
    return FaceRecognition_transformer 
