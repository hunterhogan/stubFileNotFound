from . import converter as converter, core as core, groupby as groupby, hist as hist, misc as misc, style as style, timeseries as timeseries, tools as tools
from pandas.plotting._matplotlib.boxplot import boxplot as boxplot, boxplot_frame as boxplot_frame, boxplot_frame_groupby as boxplot_frame_groupby
from pandas.plotting._matplotlib.converter import deregister as deregister, register as register
from pandas.plotting._matplotlib.hist import hist_frame as hist_frame, hist_series as hist_series
from pandas.plotting._matplotlib.misc import andrews_curves as andrews_curves, autocorrelation_plot as autocorrelation_plot, bootstrap_plot as bootstrap_plot, lag_plot as lag_plot, parallel_coordinates as parallel_coordinates, radviz as radviz, scatter_matrix as scatter_matrix
from pandas.plotting._matplotlib.tools import table as table

__all__ = ['plot', 'hist_series', 'hist_frame', 'boxplot', 'boxplot_frame', 'boxplot_frame_groupby', 'table', 'andrews_curves', 'autocorrelation_plot', 'bootstrap_plot', 'lag_plot', 'parallel_coordinates', 'radviz', 'scatter_matrix', 'register', 'deregister']

def plot(data, kind, **kwargs): ...

# Names in __all__ with no definition:
#   andrews_curves
#   autocorrelation_plot
#   bootstrap_plot
#   boxplot
#   boxplot_frame
#   boxplot_frame_groupby
#   deregister
#   hist_frame
#   hist_series
#   lag_plot
#   parallel_coordinates
#   radviz
#   register
#   scatter_matrix
#   table
