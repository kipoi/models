from kipoi.model import BaseModel
import torch


class TorchModel(BaseModel):

    def __init__(self, torch_t7_file=None, auto_use_cuda=True):

        from torch.utils.serialization import load_lua
        self.model = load_lua(torch_t7_file)
        self.model.evaluate()

        if auto_use_cuda and torch.cuda.is_available():
            self.model = self.model.cuda()
            self.use_cuda = True
        else:
            self.use_cuda = False

    def predict_on_batch(self, x):
        # numpy -> torch
        x_torch = torch.from_numpy(x)
        if self.use_cuda:
            x_torch = x_torch.cuda()
        preds_torch = self.model.forward(x_torch)

        # torch -> numpy
        if self.use_cuda:
            preds_torch = preds_torch.cpu()
        return preds_torch.data.numpy()
