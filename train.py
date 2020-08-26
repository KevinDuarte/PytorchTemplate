import config
import torch
from torch.autograd.variable import Variable
import numpy as np
from dataloader import TrainDataset, ValidationDataset, DataLoader, get_mnist_dataset
import torch.nn as nn
import torch.optim as optim
from model import Model


def get_accuracy(y_pred, y):
    y_argmax = torch.argmax(y_pred, -1)

    return torch.mean((y_argmax==y).type(torch.float))



def train(model, data_loader, criterion, optimizer):
    model.train()

    if config.use_cuda:
        model.cuda()

    losses, accs = [], []
    for i, sample in enumerate(data_loader):
        x, y = sample

        if config.use_cuda:
            x = x.cuda()
            y = y.cuda()

        optimizer.zero_grad()

        y_pred = model(x)

        loss = criterion(y_pred, y)

        acc = get_accuracy(y_pred, y)

        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        accs.append(acc.item())

        if (i + 1) % 100 == 0:
            print('Finished training %d batches. Loss: %.4f. Accuracy: %.4f.' % (i+1, float(np.mean(losses)), float(np.mean(accs))))

    print('Finished training. Loss: %.4f. Accuracy: %.4f.' % (float(np.mean(losses)), float(np.mean(accs))))

    return float(np.mean(losses)), float(np.mean(accs))


def validation(model, data_loader, criterion):
    model.eval()

    if config.use_cuda:
        model.cuda()

    losses, accs = [], []
    for i, sample in enumerate(data_loader):
        x, y = sample

        if config.use_cuda:
            x = x.cuda()
            y = y.cuda()

        y_pred = model(x)

        loss = criterion(y_pred, y)

        acc = get_accuracy(y_pred, y)

        losses.append(loss.item())
        accs.append(acc.item())

        if (i + 1) % 100 == 0:
            print('Finished validating %d batches. Loss: %.4f. Accuracy: %.4f.' % (i+1, float(np.mean(losses)), float(np.mean(accs))))

    print('Finished validation. Loss: %.4f. Accuracy: %.4f.' % (float(np.mean(losses)), float(np.mean(accs))))

    return float(np.mean(losses)), float(np.mean(accs))


def run_experiment():
    model = Model()

    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    tr_dataset = get_mnist_dataset('./data/', True, download=True)  # TrainDataset()  # A custom dataloader may be needed, in which case use TrainDataset()
    val_dataset = get_mnist_dataset('./data/', False, download=True)  # ValidationDataset() # A custom dataloader may be needed, in which case use ValidationDataset()
	
	tr_dataloader = DataLoader(tr_dataset, batch_size=config.batch_size, shuffle=True)
	val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    best_loss = 1000000
	for epoch in range(1, config.n_epochs + 1):
		print('Epoch', epoch)

		losses, _ = train(model, tr_dataloader, criterion, optimizer)

		losses, _ = validation(model, val_dataloader, criterion)

		if losses < best_loss:
			print('Model Improved -- Saving.')
			best_loss = losses

			save_file_path = os.path.join(config.save_dir, 'model_{}_{:.4f}.pth'.format(epoch, losses))
			states = {
				'epoch': epoch + 1,
				'state_dict': model.state_dict(),
				'optimizer': optimizer.state_dict(),
			}
			
			try:
				os.mkdir(config.save_dir)
			except:
				pass
			
			torch.save(states, save_file_path)
			print('Model saved ', str(save_file_path))

		print('Training Finished')


if __name__ == '__main__':
    run_experiment()
