import seaborn as sns

from . import data
from . import feature_engineering as fe
from . import hyperparameter_tune as ht
from . import evaluation as ev

# Sets the default color palette and the theme for the plots
color_palette = sns.color_palette("Blues")
sns.set_theme(style="darkgrid", palette=color_palette)




