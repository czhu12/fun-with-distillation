from torch import nn

class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2):
        super(LinearBlock, self).__init__()
        self.out_features = out_features
        self.net = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(out_features),
        )
        
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.net(x)
    
class FeedForwardModel(nn.Module):
    def __init__(self, block, layers):
        super(FeedForwardModel, self).__init__()
        self.net = nn.Sequential(*map(
                lambda x: block(*x),
                layers,
            ))
        
    def forward(self, x):
        return self.net(x)