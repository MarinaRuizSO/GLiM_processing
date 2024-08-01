import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import earthpy.spatial as es
def make_col_list(unique_vals, nclasses=None, cmap=None):
    """
    Convert a matplotlib named colormap into a discrete list of n-colors in
    RGB format.

    Parameters
    ----------
    unique_vals : list
        A list of values to make a color list from.
    nclasses : int (optional)
        The number of classes.
    cmap : str (optional)
        Colormap name used to create output list.

    Returns
    -------
    list
        A list of colors based on the given set of values in matplotlib
        format.

    Example
    -------
    >>> import numpy as np
    >>> import earthpy.plot as ep
    >>> import matplotlib.pyplot as plt
    >>> arr = np.array([[1, 2], [3, 4], [5, 4], [5, 5]])
    >>> f, ax = plt.subplots()
    >>> im = ax.imshow(arr, cmap="Blues")
    >>> the_legend = ep.draw_legend(im_ax=im)
    >>> # Get the array and cmap from axis object
    >>> cmap_name = im.axes.get_images()[0].get_cmap().name
    >>> unique_vals = list(np.unique(im.get_array().data))
    >>> cmap_colors = ep.make_col_list(unique_vals, cmap=cmap_name)

    """
    if not nclasses:
        nclasses = len(unique_vals)

    increment = 1 / (nclasses - 1)

    # Create increments to grab colormap colors
    col_index = [(increment * c) for c in range(nclasses - 1)]
    col_index.append(1.0)

    # Create cmap list of colors
    cm = plt.cm.get_cmap(cmap)

    return [cm(c) for c in col_index]
def draw_legend(im_ax, bbox=(1.05, 1), titles=None, cmap=None, classes=None, size_font=10):
    """Create a custom legend with a box for each class in a raster.

    Parameters
    ----------
    im_ax : matplotlib image object
        This is the image returned from a call to imshow().
    bbox : tuple (default = (1.05, 1))
        This is the bbox_to_anchor argument that will place the legend
        anywhere on or around your plot.
    titles : list (optional)
        A list of a title or category for each unique value in your raster.
        This is the label that will go next to each box in your legend. If
        nothing is provided, a generic "Category x" will be populated.
    cmap : str (optional)
        Colormap name to be used for legend items.
    classes : list (optional)
        A list of unique values found in the numpy array that you wish to plot.


    Returns
    ----------
    matplotlib.pyplot.legend
        A matplotlib legend object to be placed on the plot.

    Example
    -------

    .. plot::

        >>> import numpy as np
        >>> import earthpy.plot as ep
        >>> import matplotlib.pyplot as plt
        >>> im_arr = np.random.uniform(-2, 1, (15, 15))
        >>> bins = [-np.Inf, -0.8, 0.8, np.Inf]
        >>> im_arr_bin = np.digitize(im_arr, bins)
        >>> cat_names = ["Class 1", "Class 2", "Class 3"]
        >>> f, ax = plt.subplots()
        >>> im = ax.imshow(im_arr_bin, cmap="gnuplot")
        >>> im_ax = ax.imshow(im_arr_bin)
        >>> leg_neg = ep.draw_legend(im_ax = im_ax, titles = cat_names)
        >>> plt.show()
    """

    try:
        im_ax.axes
    except AttributeError:
        raise AttributeError(
            "The legend function requires a matplotlib axis object to "
            "run properly. You have provided a {}.".format(type(im_ax))
        )

    # If classes not provided, get them from the im array in the ax object
    # Else use provided vals
    if classes is not None:
        # Get the colormap from the mpl object
        cmap = im_ax.cmap.name

        # If the colormap is manually generated from a list
        if cmap == "from_list":
            cmap = ListedColormap(im_ax.cmap.colors)

        colors = make_col_list(
            nclasses=len(classes), unique_vals=classes, cmap=cmap
        )
        # If there are more colors than classes, raise value error
        if len(set(colors)) < len(classes):
            raise ValueError(
                "There are more classes than colors in your cmap. "
                "Please provide a ListedColormap with the same number "
                "of colors as classes."
            )

    else:
        classes = list(np.unique(im_ax.axes.get_images()[0].get_array()))
        # Remove masked values, could next this list comp but keeping it simple
        classes = [
            aclass for aclass in classes if aclass is not np.ma.core.masked
        ]
        colors = [im_ax.cmap(im_ax.norm(aclass)) for aclass in classes]

    # If titles are not provided, create filler titles
    if not titles:
        titles = ["Category {}".format(i + 1) for i in range(len(classes))]

    if not len(classes) == len(titles):
        raise ValueError(
            "The number of classes should equal the number of "
            "titles. You have provided {0} classes and {1} titles.".format(
                len(classes), len(titles)
            )
        )

    patches = [
        mpatches.Patch(color=colors[i], label="{lab}".format(lab=titles[i]))
        for i in range(len(titles))
    ]
    # Get the axis for the legend
    ax = im_ax.axes
    return ax.legend(
        handles=patches,
        bbox_to_anchor=bbox,
        loc=2,
        borderaxespad=0.0,
        prop={"size":size_font},
    )


def get_cbar_ax_size(ax, fig):
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    colorbar_x = 1.57/width # 1.57 inch is 4cm (size of first colorbar is 4cm)
    return colorbar_x

def CreateFigure(FigSizeFormat="default", AspectRatio=16./9., size_font = 10):
    """
    This function creates a default matplotlib figure object

    Args:
        FigSizeFormat: the figure size format according to journal for which the figure is intended
            values are geomorphology,ESURF, ESPL, EPSL, JGR, big
            default is ESURF

        AspectRatio: The shape of the figure determined by the aspect ratio, default is 16./9.

    Returns:
        matplotlib figure object

    Author: FJC
    """
    # set figure sizes (in inches) based on format
    if FigSizeFormat == "geomorphology":
        FigWidth_Inches = 6.25
    elif FigSizeFormat == "big":
        FigWidth_Inches = 16
    elif FigSizeFormat == "ESURF":
        FigWidth_Inches = 4.92
    elif FigSizeFormat == "ESPL":
        FigWidth_Inches = 7.08
    elif FigSizeFormat == "EPSL":
        FigWidth_Inches = 7.48
    elif FigSizeFormat == "JGR":
        FigWidth_Inches = 6.6

    else:
        FigWidth_Inches = 4.92126

    # Set up fonts for plots
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Liberation Sans']
    rcParams['font.size'] = size_font

    Fig = plt.figure(figsize=(FigWidth_Inches,FigWidth_Inches/AspectRatio))

    return Fig