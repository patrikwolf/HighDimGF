###################################################################################################
# Helper functions for date and time
# Python
# Harvard University
# 16.05.2024
###################################################################################################

import datetime


def get_timestamp():
    timestamp = datetime.datetime.now()
    return datetime.datetime.strftime(timestamp, '%Y-%m-%d, %H:%M:%S')


def get_dash_timestamp():
    timestamp = datetime.datetime.now()
    return datetime.datetime.strftime(timestamp, '%Y-%m-%d_%H-%M-%S')


def get_identifier(file, cloud=False):
    if cloud:
        return '[Script: ' + str(file) + ', created on: ' + get_timestamp() + r']'
    else:
        return (r'[Script: \texttt{' + str(file).replace('_', r'\_')
                + r'}, created on: \texttt{' + get_timestamp() + '}]')
