"""
This tutorial runs MNIST, CIFAR-10, and CIFAR-100 tutorials
with varying parameters.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import csv

from cleverhans_tutorials.mnist_tutorial_tf import mnist_tutorial
from cleverhans_tutorials.cifar_10_tutorial_tf import cifar_10_tutorial
from cleverhans_tutorials.cifar_100_tutorial_tf import cifar_100_tutorial

import os

def run_tutorials(epochs, batch_size, filters):
    mnist = mnist_tutorial(nb_epochs=epochs,
                           nb_filters=filters,
                           batch_size=128,
                           learning_rate=0.001,
                           clean_train=True,
                           backprop_through_attack=False)

    cifar_10 = cifar_10_tutorial(nb_epochs=epochs,
                                 nb_filters=filters,
                                 batch_size=128,
                                 learning_rate=0.001,
                                 clean_train=True,
                                 backprop_through_attack=False)

    cifar_100 = cifar_100_tutorial(nb_epochs=epochs,
                                   nb_filters=filters,
                                   batch_size=128,
                                   learning_rate=0.001,
                                   clean_train=True,
                                   backprop_through_attack=False)

    return mnist, cifar_10, cifar_100


def main(argv=None):
    results = []

    # Vary epochs
    for epoch in range(2, 11):
        (mnist, cifar_10, cifar_100) = run_tutorials(epoch, 128, 64)
        results.append({'epoch': epoch, 'filters': 64, 'type': 'mnist',
                        'clean_train_clean_eval': mnist.clean_train_clean_eval,
                        'clean_train_adv_eval': mnist.clean_train_adv_eval,
                        'adv_train_clean_eval': mnist.adv_train_clean_eval,
                        'adv_train_adv_eval': mnist.adv_train_adv_eval})
        results.append({'epoch': epoch, 'filters': 64, 'type': 'cifar_10',
                        'clean_train_clean_eval': cifar_10.clean_train_clean_eval,
                        'clean_train_adv_eval': cifar_10.clean_train_adv_eval,
                        'adv_train_clean_eval': cifar_10.adv_train_clean_eval,
                        'adv_train_adv_eval': cifar_10.adv_train_adv_eval})
        results.append({'epoch': epoch, 'filters': 64, 'type': 'cifar_100',
                        'clean_train_clean_eval': cifar_100.clean_train_clean_eval,
                        'clean_train_adv_eval': cifar_100.clean_train_adv_eval,
                        'adv_train_clean_eval': cifar_100.adv_train_clean_eval,
                        'adv_train_adv_eval': cifar_100.adv_train_adv_eval})

    # Vary filters
    for filter in (16, 32, 64, 128, 256):
        (mnist, cifar_10, cifar_100) = run_tutorials(6, 128, filter)
        results.append({'epoch': 6, 'filters': filter, 'type': 'mnist',
                        'clean_train_clean_eval': mnist.clean_train_clean_eval,
                        'clean_train_adv_eval': mnist.clean_train_adv_eval,
                        'adv_train_clean_eval': mnist.adv_train_clean_eval,
                        'adv_train_adv_eval': mnist.adv_train_adv_eval})
        results.append({'epoch': 6, 'filters': filter, 'type': 'cifar_10',
                        'clean_train_clean_eval': cifar_10.clean_train_clean_eval,
                        'clean_train_adv_eval': cifar_10.clean_train_adv_eval,
                        'adv_train_clean_eval': cifar_10.adv_train_clean_eval,
                        'adv_train_adv_eval': cifar_10.adv_train_adv_eval})
        results.append({'epoch': 6, 'filters': filter, 'type': 'cifar_100',
                        'clean_train_clean_eval': cifar_100.clean_train_clean_eval,
                        'clean_train_adv_eval': cifar_100.clean_train_adv_eval,
                        'adv_train_clean_eval': cifar_100.adv_train_clean_eval,
                        'adv_train_adv_eval': cifar_100.adv_train_adv_eval})

        with open('C:\\Users\\Sean\\Downloads\\param_comparison.csv', 'w', newline='') as f:
            field_names = ['epoch', 'filters', 'type',
                           'clean_train_clean_eval', 'clean_train_adv_eval',
                           'adv_train_clean_eval', 'adv_train_adv_eval']
            writer = csv.DictWriter(f, fieldnames=field_names)
            writer.writeheader()
            for result in results:
                writer.writerow(result)


if __name__ == '__main__':
    main()
