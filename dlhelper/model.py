import re
import tqdm
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from colored import fg, attr
from sklearn.metrics import ConfusionMatrixDisplay


class ResNet(torch.nn.Module):
    def __init__(
            self,
            num_class: int,
            arch: str = 'resnet50',
            finetune: bool = True,
            pretrained: bool = False,
            learning_rate: float = 1e-3,
            ):
        super(ResNet, self).__init__()

        # Model training components
        self.model = self.select_model(
                arch, num_class, finetune=finetune, pretrained=pretrained)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, patience=5, threshold=1e-1)

        # Move to device
        self.device = torch.device(
                "cuda:0" if torch.cuda.is_available else "cpu")
        self.model = self.model.to(self.device)

        # Initialize
        self.tot_size, self.tot_loss, self.tot_correct = 0, 0, 0

        # Debug/visualize
        self.conf_mtx = np.zeros([num_class, num_class])

    def select_model(self, arch, num_class, finetune, pretrained):
        print(f"[ INFO ] Use torchvision pretrain: {pretrained}")
        model = torchvision.models.__dict__[arch](pretrained=pretrained)
        freeze = not finetune
        for param in model.parameters():
            param.requires_grad = freeze
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, num_class)
        return model

    def train_model(self, dataloader):
        """
        Return loss, accuracy
        """
        self._reset_status()
        # NOTE: model should be eval when linear finetune
        self.model = self.model.train()

        pbar = tqdm.tqdm(dataloader, ncols=77, desc="==> Training")
        for image, label in pbar:
            image = image.to(self.device)
            label = label.to(self.device)

            output = self.model(image)
            pred = torch.argmax(output, dim=1)
            loss = self.criterion(output, label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self._update_status(pred, label, loss)

        train_loss, train_acc = self._cal_metric()
        self.scheduler.step(train_loss)
        last_lr = self.scheduler._last_lr[0]
        print(f"=> learning rate now: {last_lr:.2e}")

        return train_loss, train_acc

    @torch.no_grad()
    def eval_model(self, dataloader):
        """
        Return loss, accuracy
        """
        self._reset_status()
        self.model = self.model.eval()

        pbar = tqdm.tqdm(dataloader, ncols=77, desc="==> Validating")
        for image, label in pbar:
            image = image.to(self.device)
            label = label.to(self.device)

            output = self.model(image)
            pred = torch.argmax(output, dim=1)
            loss = self.criterion(output, label)

            self._update_status(pred, label, loss)

        return self._cal_metric()

    def load_state_dict(self, state_dict):
        model_state_dict = self.model.state_dict()
        pretrain_keys = list(state_dict.keys())
        match_dict = dict()
        for model_key in list(model_state_dict.keys()):
            # NOTE There may be better match methods
            model_key_component = model_key.split('.')
            model_key_component[-2] += '.*\\'
            re_model_key = '.'.join(model_key_component)

            for pretrain_key in pretrain_keys:
                if re.search(re_model_key, pretrain_key) is not None:
                    match_dict[model_key] = state_dict[pretrain_key]
                    break

        msg = self.model.load_state_dict(match_dict, strict=False)
        print(f"[ INFO ] Missing keys: {msg.missing_keys}")
        return

    def _reset_status(self):
        self.tot_loss = 0.0
        self.tot_correct = 0.0
        self.tot_size = 0.0
        return

    @torch.no_grad()
    def _update_status(self, pred, gt, loss):
        self.tot_size += pred.shape[0]
        self.tot_loss += loss * pred.shape[0]
        self.tot_correct += torch.sum(pred == gt)
        for p, g in zip(pred, gt):
            self.conf_mtx[p][g] += 1
        return

    @torch.no_grad()
    def _cal_metric(self):
        avg_acc = self.tot_correct / self.tot_size
        avg_loss = self.tot_loss / self.tot_size
        print(f"=> {fg(189)}loss: {avg_loss:.3f}, "
              f"{fg(177)}accuracy: {avg_acc:.3%} {attr(0)}")
        # heat_map = ConfusionMatrixDisplay(self.conf_mtx)
        # heat_map.plot(include_values=False)
        # plt.savefig("confusion.jpg")
        return avg_loss, avg_acc


if __name__ == "__main__":
    model = ResNet()
