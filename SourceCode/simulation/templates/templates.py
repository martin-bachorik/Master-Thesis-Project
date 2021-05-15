from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")


class Template:
    def __init__(self):
        """ Serves as a tool for adjusting the figure shape fitting Master Thesis

        """
        # create a new enviromental variable: C:\Users\Martin\AppData\Local\Programs\MiKTeX 2.9\miktex\bin\x64
        # import matplotlib as mpl
        # mpl.rcParams.keys()
        # MAKE FIGURE SETTINGS
        ################################################################################################################
        # Use the seborn style
        # But with fonts from the document body
        # mpl.use('pgf')
        # plt.style.use('seaborn')
        # plt.style.use('seaborn-paper')

        plt.rcParams.update({
            "font.family": "serif",  # use serif/main font for text elements
            # "font.sans-serif": ["Helvetica"],
            "text.usetex": True,  # use inline math for ticks
            "pgf.rcfonts": False,  # don't setup fonts from rc parameters
            "patch.linewidth": 0.3,  # border width
            "patch.antialiased": True,

            "lines.linewidth": 1,
            "lines.markersize": 3,
            "lines.antialiased": True,

            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "axes.labelpad": 4,  # distance between the axis and label
            "axes.titlepad": 5,  # distance between the axis and title
            "axes.formatter.limits": [-5, 4],  # set limits to use scientific notation
            "axes.linewidth": 0.2,
            "axes.formatter.use_mathtext": True,

            # "legend.borderaxespad": 0.0,
            "legend.columnspacing": 0.5,
            "legend.fancybox": False,

            "grid.linewidth": 0.2,
            "grid.color": "bfbfbf",  # change grid line color to lighter gray, so it looks like in Matlab

            "xtick.direction": 'in',
            "xtick.labelsize": 8,
            "xtick.major.width": 0.2,
            "xtick.top": True,

            "ytick.direction": 'in',
            "ytick.labelsize": 8,
            "ytick.major.width": 0.2,
            "ytick.right": True,

            "savefig.bbox": 'tight',  # remove white spaces around the figure
            "savefig.pad_inches": 0.03,  # absolutely no white space
        })
        # plt.rc('font', family='sans-serif')
        plt.rc('text', usetex=True)
        plt.rc('pgf', texsystem='xelatex')  # or luatex, xelatex...


# Master Thesis: 371.30264 PT
def set_size(width_pt=371.30264):

    """Set figure dimensions according to the width.

    Args:
        width_pt: Doc width in points(pt)

    Returns:
            tuple: Figure dimensions in inches.
    """
    plt.rcParams.update({
        "savefig.pad_inches": 0.06,  # absolutely no white space
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,

    })
    width_pt = width_pt  # Width of figure (in pts)
    inch_pt = 1 / 72.27  # Convert from pt to inches
    golden_ratio = (5**.5 - 1) / 2  # Golden ratio
    width_in = width_pt * inch_pt  # Figure width in inches
    height_in = width_in * golden_ratio  # Figure height in inches
    return width_in, height_in


def set_size_orig(width_pt=371.30264):

    """Set figure dimensions according to the width.

    Args:
        width_pt: Doc width in points(pt)

    Returns:
            tuple: Figure dimensions in inches.
    """
    # Template()
    width_pt = width_pt  # Width of figure (in pts)
    inch_pt = 1 / 72.27  # Convert from pt to inches
    golden_ratio = (5**.5 - 1) / 2  # Golden ratio
    width_in = width_pt * inch_pt  # Figure width in inches
    height_in = width_in * golden_ratio  # Figure height in inches
    return width_in, height_in


def set_size_long():
    # Template()
    width_in = 5.137714681057147
    height_in = 1.8
    return width_in, height_in
