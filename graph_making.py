import os
import matplotlib.pyplot as plt


if __name__=="__main__":
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    with open(os.getcwd() + "/results_alexnet.txt", 'r') as f:
        change = 0
        lines = f.readlines()
        new_lines = [line[:-1] for line in lines]

        for line in new_lines:
            split_line = line.split(',')
            if change % 2 == 0:
                train_loss.append(float(split_line[0]))
                train_acc.append(float(split_line[1]))
            else:
                val_loss.append(float(split_line[0]))
                val_acc.append(float(split_line[1]))

            change = change + 1

    epochs = list(range(0, len(train_loss)))

    plt.plot(epochs, train_acc, label="train")
    plt.plot(epochs, val_acc, label="val")
    plt.title("Accuracy of the model - Alexnet")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    #####################

    plt.plot(epochs, train_loss, label="train")
    plt.plot(epochs, val_loss, label="val")
    plt.title("Loss of the model - Alexnet")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

