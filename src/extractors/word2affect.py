import torch
from torch import nn
from transformers import AutoModel
# git clone https://huggingface.co/hplisiecki/word2affect_english with LFS installed
class CustomModel(torch.nn.Module):
    def __init__(self, model_path, dropout=0.1, hidden_dim=768):
        super().__init__()
        self.metric_names = ['valence', 'arousal', 'dominance', 'aoa', 'concreteness']
        self.dropout_rate = dropout
        self.hidden_dim = hidden_dim

        self.bert = AutoModel.from_pretrained(model_path)

        for name in self.metric_names:
            setattr(self, name, nn.Linear(hidden_dim, 1))
            setattr(self, 'l_1_' + name, nn.Linear(hidden_dim, hidden_dim))

        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_rate)
        self.sigmoid = nn.Sigmoid()

    def save_pretrained(self, save_directory):
        self.bert.save_pretrained(save_directory)
        torch.save(self.state_dict(), f'{save_directory}/pytorch_model.bin')

    @classmethod
    def from_pretrained(cls, model_dir, dropout=0.2, hidden_dim=768):
        model = cls(model_dir, dropout, hidden_dim)
        state_dict = torch.load(f'{model_dir}/pytorch_model.bin', map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        return model

    def forward(self, *args):
        _, x = self.bert(*args, return_dict=False)
        output = self.rate_embedding(x)
        return output

    def rate_embedding(self, x):
        output_ratings = []
        for name in self.metric_names:
            first_layer = self.relu(self.dropout(self.layer_norm(getattr(self, 'l_1_' + name)(x) + x)))
            second_layer = self.sigmoid(getattr(self, name)(first_layer))
            output_ratings.append(second_layer)

        return output_ratings