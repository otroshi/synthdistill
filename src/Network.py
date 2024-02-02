import torch
import timm

class LightNetwork(torch.nn.Module):
    def __init__(self, model_name='tinynet_a'):
        super().__init__()
        self.cnn = timm.create_model(model_name=model_name,
                                    pretrained=True,
                                    num_classes=0,)

        self.fc = torch.nn.Linear(self.cnn.num_features, 512)
    def forward(self, input):
        feat = self.cnn(input)
        out = self.fc(feat)
        return out

    def get_embedding(self, input):
        return self.cnn(input)