from random import shuffle
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

class TrainingConfig:
    learning_rate = 4e-3
    epsilon = 1e-8
    betas = (0.9, 0.999)
    weight_decay = 0.05
    max_epochs = 300

    num_workers = 8
    batch_size = 64
    shuffle = True
    pin_memory = True

    ckpt_path=None
    logdir = "RegNet"

    device = "cuda"

    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)

class Trainer:
    def __init__(self, model, train_dataset, test_dataset, configs: TrainingConfig) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.configs = configs

        if self.configs.device == "cuda":
            self.device = torch.cuda.current_device()
            self.model = DataParallel(self.model).to(self.device)

    def save_checkpoint(self, msg):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        torch.save(raw_model.state_dict(), self.configs.ckpt_path[:-3]+msg+".pt")
        print("Model Saved")

    def load_checkpoint(self, path):
        self.model.load_state_dict(torch.load(path))
        print("Model loaded from", path)

    def train(self):
        model, config = self.model, self.configs
        raw_model = model.module if hasattr(model, "module") else model
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr = config.learning_rate,
                                      weight_decay = config.weight_decay,
                                      betas = config.betas,
                                      eps = config.epsilon)

        def run_epoch(split):
            is_train = split == "train"
            if is_train:
                model.train()
            else:
                model.eval()


            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, batch_size = config.batch_size, num_workers=config.num_workers,
                                pin_memory=config.pin_memory, shuffle=config.shuffle)

            losses = []
            accuracies = []
            correct = 0
            num_smamples = 0

            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, batch in pbar:
                x,y = batch
                num_smamples += x.size(0)

                x = x.to(self.device)
                y = y.to(self.device)

                logits, loss = model(x, y)

                with torch.no_grad():
                    predictions = torch.argmax(logits, dim=1) 
                    correct += predictions.eq(y).sum().item()
                    accuracy = correct/num_smamples
                    accuracies.append(accuracy)
                    losses.append(loss.item())

                if is_train:
                    model.zero_grad()
                    loss.backward()
                    optimizer.step()

                    pbar.set_description(f"Epoch: {epoch+1} it: {it+1} | loss: {loss.item():.5f} accuracy: {accuracy:.5f}")
                
                if not is_train:
                    test_loss = float(np.mean(losses))
                    test_accuracy = float(np.mean(accuracies))
                    print(f"Test loss: {test_loss} accuracy: {test_accuracy}")
                    return test_loss

        best_loss = float('inf')
        test_loss = float('inf')

        for epoch in range(self.configs.max_epochs):
            run_epoch('train')

            if self.test_dataset is not None:
                test_loss = run_epoch('test')

            good_model = self.test_dataset is None or test_loss < best_loss
            if self.configs.ckpt_path is not None and good_model:
                best_loss = test_loss
                self.save_checkpoint(f"epoch-{epoch+1}")





