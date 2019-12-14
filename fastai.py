from fastai.vision import *
from fastai.callbacks.hooks import *
from fastai.utils.mem import *
import PIL
import pandas as pd
from pathlib import Path


# This code was adapted from a fastai tutorial for image segmentation
# Original notebook can be found here:
# https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson1-pets.ipynb

path = Path('C:/Users/RRG01/OneDrive - Quantumscape Corporation/fa-tears/fastai')
path_train = path / 'train'
path_test = path / 'test'
model_name = "fa-tear-detection"rint("Finding sample mask")

bs = 4

data = ImageDataBunch.from_folder(path, train=path_train, valid_pct=0.2, bs=bs)

learn = cnn_learner(data, models.resnet34, metrics=error_rate)

learn.fit_one_cycle(4)

lr_find(learn)
learn.recorder.plot()

# Was not able to extract useful information for learning rate from lr_finder

interp = ClassificationInterpretation.from_learner(learn)
losses, idx = interp.top_losses()
interp.plot_top_losses(4, figsize=(10,10))

# Top losses were all predicted no-tears and actual tears (false negatives)

learn.unfreeze()  # Now we are training all of the layers, instead of just the final layers
learn.fit_one_cycle(2, max_lr=1e-6)

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(10,10), dpi=60)

# Now we have more false-positives than false negatives, but our loss
# is still highest for false negatives. This is because the predicted
# probability for the false negatives is much lower, ie the model is very
# confident that the image does not have a defect
