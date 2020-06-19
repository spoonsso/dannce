"""Entrypoints for dannce training and prediction."""
from dannce.interface import com_predict, com_train, dannce_predict, dannce_train
import sys


def com_predict_cli():
    com_predict(sys.argv[1])


def com_train_cli():
    com_train(sys.argv[1])


def dannce_predict_cli():
    dannce_predict(sys.argv[1])


def dannce_train_cli():
    dannce_train(sys.argv[1])
