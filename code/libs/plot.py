import sys

def text_ax(ax,
            text, 
            loc):
    """
        Build 'Fixed parameter' text axis
        loc: bottom left coordinate of the text area bounding box
    """
    
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    ax.set_facecolor('white')

    ax.text(0,
            0, 
            text, 
            horizontalalignment='left', 
            verticalalignment='bottom')
    
    return ax