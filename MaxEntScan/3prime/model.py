import os
import inspect

filename = inspect.getframeinfo(inspect.currentframe()).filename
this_path = os.path.dirname(os.path.abspath(filename))


# attach template to pythonpath
import sys
sys.path.append(os.path.join(this_path, "../template"))

from model_template import MaxEntModel


class MaxEntModelSpec(MaxEntModel):

    def __init__(self):
        super(MaxEntModelSpec, self).__init__(side='3prime')
