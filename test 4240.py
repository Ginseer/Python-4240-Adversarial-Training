# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 15:24:51 2023

@author: colto
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from progress.bar import Bar

def generate_data():
    # generate some sample data
    cifar10_data = np.random.rand(10)
    mnist_data = np.random.rand(5)
    return cifar10_data, mnist_data

def adversarial_training(data, epochs):
    # simulate adversarial training
    before_acc = np.random.rand()
    with Bar('Training', max=epochs) as bar:
        for i in range(epochs):
            bar.next()
            time.sleep(0.1)
    after_acc = np.random.rand()
    return before_acc, after_acc

def generate_pie_chart(labels, data, title):
    # Check that the length of the labels list matches the length of the data list
    if len(labels) != len(data):
        raise ValueError("The length of the labels list must match the length of the data list.")
    
    # Calculate the percentage of correct identifications for each class
    percentages = [100.0 * value / sum(data) for value in data]
    labels = [f'{label} ({percentages[i]:.1f}%)' for i, label in enumerate(labels)]
    
    # Generate the pie chart
    fig, ax = plt.subplots()
    wedges, _, _ = ax.pie(data, labels=labels, autopct='%1.1f%%', textprops=dict(color="w"))
    ax.set_title(title)
    plt.legend(wedges, labels, loc="best", bbox_to_anchor=(1, 0, 0.5, 1))

    plt.show()

def generate_bar_chart(cifar10_data, mnist_data):
    # generate a bar chart showing improvement
    improvement = [cifar10_data[i] - mnist_data[i] for i in range(5)]
    colors = ['green' if i >= 0 else 'red' for i in improvement]
    plt.bar(range(5), improvement, color=colors)
    plt.title('Improvement')
    plt.xlabel('Data Point')
    plt.ylabel('Improvement')
    plt.show()

# generate some data
cifar10_data, mnist_data = generate_data()

# adversarial training
epochs = 10
before_acc_cifar10, after_acc_cifar10 = adversarial_training(cifar10_data, epochs)
before_acc_mnist, after_acc_mnist = adversarial_training(mnist_data, epochs)

# generate pie charts
generate_pie_chart(['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5'], cifar10_data[:5], 'Cifar-10')
generate_pie_chart(['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5'], mnist_data, 'MNIST')

# print accuracy results
print('Cifar-10 accuracy before adversarial training:', before_acc_cifar10)
print('Cifar-10 accuracy after adversarial training:', after_acc_cifar10)
print('MNIST accuracy before adversarial training:', before_acc_mnist)
print('MNIST accuracy after adversarial training:', after_acc_mnist)

# generate bar chart showing improvement
generate_bar_chart(cifar10_data, mnist_data)