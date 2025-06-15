import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Use VSCode renderer for interactive display in VSCode environments
# Other options: 'browser', 'notebook', 'iframe', etc.
pio.renderers.default = 'browser'

def shorten(string):
    if 'e' in string:
        left = string.split('e')[0][:7]
        right = string.split('e')[1][:7]
        return left + 'e' + right
    else:
        if '.' in string:
            count = 0
            for i in range(len(string.split('.')[1])):
                if string[i] == '0':
                    count += 1
            return string[:count+5]
        else:
            return string[:7]
        
def give_title(image, title = '', idx = '', min_max = True):    
    if min_max:
        min_val_orig = np.min(image)
        max_val_orig = np.max(image)
        txt_min_val = '('+shorten(str(min_val_orig))+', '
        txt_max_val = shorten(str(max_val_orig))+')'
    else:
        txt_min_val = ''
        txt_max_val = ''    
    title = str(int(idx) + 1) if title == '' else title
    return title+'\n'+txt_min_val+txt_max_val if min_max else title

def give_titles(images, titles = [], min_max = True):
    titles = [titles] if type(titles) is not list else titles
    if len(titles) <= len(images):
        titles = [give_title(images[i], title = titles[i], idx=i, min_max = min_max) for i in range(len(titles))]
        n_for_rest = np.arange(len(titles), len(images))
        titles.extend([give_title(images[i], idx=i, min_max = min_max) for i in n_for_rest])
    else:
        titles = [give_title(images[i], title = titles[i], idx=i, min_max = min_max) for i in range(len(images))]
    return titles

def get_row_col(images, show_all = False, images_per_row = 5):
    if show_all:
        rows = int(np.sqrt(len(images)))
        cols = int(np.sqrt(len(images)))
        return rows, cols + (len(images) - rows*cols)//rows
    
    if len(images) == 1:
        rows = 1
        cols = 1
    elif len(images) <= images_per_row:
        rows = 1
        cols = len(images)
    else:
        rows = len(images)//images_per_row
        cols = images_per_row
        if rows*cols < len(images):
            rows += 1
    
    return rows, cols

def chose_fig_size(images, rows = None, cols = None, show_all = False, add_length = None, images_per_row = 5, fig_size = None):
    (rows, cols) = get_row_col(images, show_all, images_per_row) if rows is None or cols is None else (rows, cols)
    shape = images[0].shape
    if fig_size is not None:    
        fig_size = fig_size
    else:
        if shape[0] > 260:
            fig_size = (shape[1]*cols/100+1, shape[0]*rows/100)
        elif shape[0] > 100 and shape[0] <= 260:
            fig_size = (shape[1]*cols/50+1, shape[0]*rows/50)
        else:
            fig_size = (shape[1]*cols/25+1, shape[0]*rows/25)
        if add_length is None:
            add_length = 5
            fig_size = (fig_size[0]+add_length, fig_size[1])
    return fig_size, rows, cols
            
def visualize(
    data,
    x=None,
    y=None,
    kind='scatter',
    color=None,
    size=None,
    facet_row=None,
    facet_col=None,
    title=None,
    template='plotly_white',
    profile=False,
    profile_axis='row',
    profile_index=None,
    images_per_row=3,
    **kwargs
):
    """ this version will be used for visualization of images, 
    it will use plotly to visualize images, and it will use the same interface as the previous version. 
    Use profile axis to profile the images,
    profile_index to specify the index of the image to profile,
    images_per_row to specify the number of images per row.
    This function supports both image display and other kinds of plots using Plotly Express.
    For image display (`kind='image'`), it shows one or multiple numpy arrays side by side.
    Internally uses `plotly.express.imshow` for richer rendering and correct orientation.
    Parameters:
        data: Data to visualize, can be a list of numpy arrays for images or a DataFrame for other kinds of plots.
        x: Column name or array-like for x-axis (optional).
        y: Column name or array-like for y-axis (optional).
        kind: Type of plot to create ('scatter', 'line', 'bar', 'histogram', 'box', 'violin', 'pie', 'heatmap', 'image').
        color: Column name or array-like for color encoding (optional).
        size: Column name or array-like for size encoding (optional).
        facet_row: Column name for row facets (optional).
        facet_col: Column name for column facets (optional).
        title: Title of the plot (optional).
        template: Plotly template to use for styling (default is 'plotly_white').
        images_per_row: Number of images to display per row when `kind='image'`.
        **kwargs: Additional keyword arguments passed to Plotly functions.
    """
    if kind == 'image':
        # Handle image display
        images = data if isinstance(data, list) else [data]
        names = kwargs.get('names', [None] * len(images))
        cols = min(len(images), images_per_row)
        rows = (len(images) + cols - 1) // cols
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=names)
        
        for idx, img in enumerate(images):
            tmp = px.imshow(
                img,
                origin='lower',
                color_continuous_scale='gray',
                template=template
            )
            trace = tmp.data[0]
            fig.add_trace(trace, row=(idx // cols) + 1, col=(idx % cols) + 1)
            fig.update_xaxes(visible=False, row=(idx // cols) + 1, col=(idx % cols) + 1)
            fig.update_yaxes(visible=False, row=(idx // cols) + 1, col=(idx % cols) + 1)
        
        fig.update_layout(title=title or '', template=template)
    else:
        # Use Plotly Express for other kinds of plots
        fig = visualize(
            data=data,
            x=x,
            y=y,
            kind=kind,
            color=color,
            size=size,
            facet_row=facet_row,
            facet_col=facet_col,
            title=title,
            template=template,
            **kwargs
        )
    # Update layout for better spacing
    fig.update_layout(margin=dict(l=40, r=40, t=60, b=40))
    return fig
# Example usage
if __name__ == '__main__':
    from skimage import data as skdata

    cam = skdata.camera()
    horse = skdata.horse()
    fig = visualize([cam]*10, kind='image', names=['Cameraman', 'Horse'], title='Sample Images', images_per_row=3)
    fig.show()  # opens in browser