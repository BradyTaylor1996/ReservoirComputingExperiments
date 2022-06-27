import torch
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class trainer(object):
    def __init__(self, model, optimizer, criterion, save_path, epochs):
        self.model = model
        self.criterion = criterion
        self.epochs = epochs
        self.optimizer = optimizer
        self.save_path = save_path

    def train(self, train_loader, test_loader):
        best_score = 999
        best_epoch = 0
        # bias1_list = []
        # bias2_list = []
        loss_list = []
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            batch_count = 0
            start = time.time()
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                scores = self.model(inputs)
                loss = self.criterion(scores, targets)
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                batch_count += 1
            # bias1, bias2 = self.model.bias_value()
            # bias1, bias2 = torch.mean(torch.abs(bias1)), torch.mean(torch.abs(bias2))
            # print(bias1.item())
            # bias1_list.append(bias1.item())
            # bias2_list.append(bias2.item())
            print('epoch', epoch)
            print('train CE loss: %f' % (epoch_loss / batch_count))

            self.model.eval()
            epoch_loss = 0
            batch_count = 0
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                scores = self.model(inputs)
                loss = self.criterion(scores, targets)
                epoch_loss += loss.item()
                batch_count += 1

            print('epoch', epoch)
            print('test CE loss: %f' % (epoch_loss / batch_count))
            loss_list.append(epoch_loss / batch_count)
            end = time.time()
            print('epoch time: ', (end-start))

            if epoch_loss / batch_count < best_score:
                self.save_model()
                best_score = epoch_loss / batch_count
                best_epoch = epoch
            if epoch - best_epoch >= 20:
                print(loss_list)
                # print(bias1_list)
                # print(bias2_list)
                break

    def save_model(self):
        print('------ SAVE MODEL CKPT ------')
        torch.save(self.model.state_dict(), self.save_path)
