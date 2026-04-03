import numpy as np
import scipy
import matplotlib
import pandas as pd
import tensorflow as tf
import onnxruntime as ort

print('numpy:', np.__version__)
print('scipy:', scipy.__version__)
print('matplotlib:', matplotlib.__version__)
print('pandas:', pd.__version__)
print('tensorflow:', tf.__version__)
print('onnxruntime:', ort.__version__)
print()
print('TF devices:', tf.config.list_physical_devices())
print('ORT providers:', ort.get_available_providers())
