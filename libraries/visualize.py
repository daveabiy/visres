from libraries.utils import *
from libraries.torch_utils import *
import matplotlib
from matplotlib import pyplot as plt
import imageio
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from skimage.data import shepp_logan_phantom, astronaut, camera, horse
from skimage.color import rgb2gray
import skimage.io as io
from skimage.draw import polygon
from PIL import Image, ImageDraw, ImageFont

# ['GTK3Agg', 'GTK3Cairo', 'GTK4Agg', 'GTK4Cairo', 'MacOSX', 'nbAgg', 'QtAgg', 'QtCairo', 'Qt5Agg', 'Qt5Cairo', 'TkAgg', 'TkCairo', 'WebAgg', 'WX', 'WXAgg', 'WXCairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template']
# matplotlib.use('pdf')
import colorsys
named_colors = matplotlib.colors.get_named_colors_mapping()
def is_strong_color(color_name):
    """Returns True if the color is considered strong (not light or pastel)."""
    rgb = matplotlib.colors.to_rgb(color_name)  # Convert to RGB
    h, l, s = colorsys.rgb_to_hls(*rgb)         # Convert to HLS
    return l < 0.5  # Keep colors with low luminance (strong/dark colors)
strong_colors = sorted([name for name in named_colors if is_strong_color(named_colors[name])])
       
bbox_to_anchors = {
    'lower center': (0.5,-0.17),
    'upper center': (0.5,1.1),
    'upper right': (1.3, 1.0),
    'upper left': (0.0, 1.0),
    'lower right': (1.3,-0.10),
    'lower left': (-0.3,-0.17),
    'center': (0.5,0.5),
    'upper': (0.5,1.1),
    'lower': (0.5,-0.17),
    'right': (1.35, 0.0),
    'center right': (0.9, 0.6),
    'center left': (0,0.5),
    'None': (None, None)
}

locs = {
    'lower center': 'lower center',
    'upper center': 'upper center',
    'upper right': 'upper right',
    'upper left': 'upper left',
    'lower right': 'lower right',
    'lower left': 'lower left',
    'center': 'center',
    'upper': 'upper',
    'lower': 'lower',
    'right': 'right',
    'center right': 'center right',
    'center left': 'center left',
    'None': None
}
def insert_axins(ax, loc = 'upper right', width = "50%", height = "5%"):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins1 = inset_axes(
        ax,
        width=width,  # width: 50% of parent_bbox width
        height=height,  # height: 5%
        loc=loc,
    )
    axins1.xaxis.set_ticks_position("bottom")
    return axins1

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

def show_image_with_zoomed_in_part(image, left = None, right = None, buttom = None, top = None, vmode= 'show', plot_color = 'blue', color = 'red', text = None, second = None, third=None, save_name = None, save_path = None):
    #make a figure with size 20x20
    fig = plt.figure(figsize=(20,20))
    #add a subplot
    ax = fig.add_subplot(111)
    #show the image
    ax.imshow(image)
    if second == None:
        if [left, right, buttom, top] == [None, None, None, None]:
            #25% of the image at the center
            left = image.shape[1]//4
            right = 3*image.shape[1]//4
            buttom = image.shape[0]//4
            top = 3*image.shape[0]//4

        if vmode == 'show':
            ax.add_patch(plt.Rectangle((left, buttom), right - left, top - buttom, edgecolor=color, lw=2, facecolor='none'))
            axins = ax.inset_axes([0.7, 0.7, 0.3, 0.3])
            axins.imshow(image[buttom:top, left:right])
        elif vmode == 'plot':
            #add a line patch
            ax.add_patch(plt.Rectangle((left, buttom+(top-buttom)//2), right - left, 1, edgecolor=plot_color, lw=2, facecolor='none'))
            axins = ax.inset_axes([0.7, 0.7, 0.3, 0.3])
            axins.plot(np.arange(left, right), image[buttom+(top-buttom)//2, left:right], color=plot_color)
            axins.axis('off')
        else:
            """both"""
            #add a line patch
            ax.add_patch(plt.Rectangle((left, buttom+(top-buttom)//2), right - left, 1, edgecolor=plot_color, lw=2, facecolor='none'))
            axins = ax.inset_axes([0.7, 0.7, 0.3, 0.3])
            axins.plot(np.arange(left, right), image[buttom+(top-buttom)//2, left:right], color=plot_color)
            axins.axis('off')
            #add a rectangle patch
            ax.add_patch(plt.Rectangle((left, buttom), right - left, top - buttom, edgecolor=color, lw=2, facecolor='none')        )
            axins = ax.inset_axes([0, 0, 0.3, 0.3])
            axins.axis('off')
            ax.axis('off')
            axins.imshow(image[buttom:top, left:right])
    else:
        axins = ax.inset_axes(0.7, 0.7, 0.3, 0.3)
        axins.imshow(second)
        axins.axis('off')
    #write a text on the zoomed in part
    axins.text(0.8, 0.8, text, fontsize=8, ha='left', va='bottom', color='red')
    #put a third image that the user can add
    if third is not None:
        axins2 = ax.inset_axes([0.7, 0.7, 0.1, 0.1])
        # print(third)
        axins2.imshow(third)
        axins2.axis('off')
    #remove axis
    axins.axis('off')
    ax.axis('off')
    plt.show()

    if save_path is not None:
        save_path = os.getcwd() + '/plots/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if save_name is None:
            save_name = 'zoomed_in_image'
        save_path = save_path + save_name + '.png'
        plt.savefig(save_path)
                      
def val_from_images(image, type_of_image = None):
    if type_of_image is None:
        type_of_image = type(image)
    if 'ndarray' in str(type_of_image):
        if len(image.shape) == 2:
            val = image
        elif len(image.shape) == 3:
            val = [image[j,:,:] for j in range(len(image))]
        else:
            val = [image[j,0,:,:] for j in range(len(image))]
    elif 'Tensor' in str(type_of_image):
        image = tensor_to_np(image)
        if type(image) is not list:
            if len(image.shape) == 2:
                val = image
            elif len(image.shape) == 3:
                val = [image[j,:,:] for j in range(len(image))]
            elif len(image.shape) == 4:
                val = [image[j,0,:,:] for j in range(len(image))]
            elif len(image.shape) == 1:
                val = image
        else:
            val = image
    elif 'jax' in str(type_of_image):
        #jax to numpy
        image = np.array(image)
        if len(image.shape) == 2:
            val = image
        elif len(image.shape) == 3:
            val = [image[j,:,:] for j in range(len(image))]
        elif len(image.shape) == 4:
            val = [image[j,0,:,:] for j in range(len(image))]
        elif len(image.shape) == 1:
            val = image
        else:
            val = image
    elif type_of_image == 'str':
        val = io.imread_collection(image)
    elif 'collection' in str(type_of_image):
        val = image
    elif 'list' in str(type_of_image):
        val = [val_from_images(image, type_of_image = type(image)) for image in image]
    else:
        assert False, "type_of_image is not nd.array, list or torch.Tensor"
    return val
    
def convert_images(images, idx = None):
    if idx is not None:
        images = [images[i] for i in idx]
    if type(images) is list:
        vals = [val_from_images(image, type_of_image = type(image)) for image in images]
        # vals = [torch_reshape(image) for image in images]
        # vals = [tensor_to_np(image) for image in vals]
  
        for i, val in enumerate(vals):
            if type(val) is list:
                [vals.append(val[j]) for j in range(len(val))]
                vals.pop(i)
        images = vals
    else:
        images = val_from_images(images, type_of_image = type(images))
    for i, val in enumerate(images):
        if type(val) is list:
            [images.append(val[j]) for j in range(len(val))]
            images.pop(i)
    return images

def plot_func(ax, plot_axis, image, plot_color, add_patch, insert_axes = True, cmap = 'gray', axis = 'off', rectangle = [0.7, 0.7, 0.3, 0.3], axin_axis = 'off',  **kwargs):
    shape = image.shape
    if insert_axes:
        axin = ax.inset_axes(rectangle)
    else:
        axin = ax
    if add_patch:
        if plot_axis == 'half':
            ax.add_patch(plt.Rectangle((1, shape[0]//2), shape[1] - 1, 1, edgecolor=plot_color, lw=2, facecolor='none'))
            axin.plot(np.arange(1, shape[1]), image[shape[0]//2, 1:shape[1]], color=plot_color)
        elif plot_axis == 'vertical':
            ax.add_patch(plt.Rectangle((shape[1]//2, 1), 1, shape[0] - 1, edgecolor=plot_color, lw=2, facecolor='none'))
            axin.plot(np.arange(1, shape[0]), image[1:shape[0], shape[1]//2], color=plot_color)
        elif plot_axis == 'diagonal':
            initial_point = (1, 1)
            final_point = (shape[1], shape[0])
            ax.add_patch(plt.Arrow(initial_point[0], initial_point[1], final_point[0] - initial_point[0], final_point[1] - initial_point[1], edgecolor=plot_color, lw=2, facecolor='none'))
            axin.plot(np.arange(np.diag(image)[1:].shape[0]), np.diag(image)[1:], color=plot_color)
            
        elif plot_axis == 'diagonal_2':
            initial_point = (1, shape[0])
            final_point = (shape[1], 1)
            ax.add_patch(plt.Arrow(initial_point[0], initial_point[1], final_point[0] - initial_point[0], final_point[1] - initial_point[1], edgecolor=plot_color, lw=2, facecolor='none'))
            axin.plot(np.arange(np.diag(np.fliplr(image))[1:].shape[0]), np.diag(np.fliplr(image))[1:], color=plot_color)
        else:
            ax.add_patch(plt.Rectangle((1, plot_axis), shape[1] - 1, 1, edgecolor=plot_color, lw=2, facecolor='none'))
            axin.plot(np.arange(1, shape[1]), image[plot_axis, 1:shape[1]], color=plot_color)
    else:
        if plot_axis == 'half':
            axin.plot(np.arange(1, shape[1]), image[shape[0]//2, 1:shape[1]], color=plot_color)
        elif plot_axis == 'vertical':
            axin.plot(np.arange(1, shape[0]), image[1:shape[0], shape[1]//2], color=plot_color)
        elif plot_axis == 'diagonal':
            axin.plot(np.arange(1, shape[0]), np.diag(image)[1:], color=plot_color)
        elif plot_axis == 'diagonal_2':
            axin.plot(np.arange(1, shape[0]), np.diag(np.fliplr(image))[1:], color=plot_color)
        else:
            axin.plot(np.arange(1, shape[1]), image[plot_axis, 1:shape[1]], color=plot_color)
            
    if axis == 'off' or axis == False:
        ax.axis('off')
    else:
        if 'legend_size' in kwargs.keys():
            ax.legend(loc = 'upper center', fontsize = kwargs['legend_size'])
        if 'title' in kwargs.keys():
            if kwargs['title'] != 'no_title':
                ax.set_title(kwargs['title'])
        else:
            ax.set_title('Plot Profile')

    if axin_axis == 'off' or axin_axis == False:
        axin.axis('off')
        
    #adjust the axis size
    ax.axis('tight')
    axin.axis('tight')
    
    return ax

def chose_fig(images, rows = None, cols = None, show_all = False, add_length = None, images_per_row = 5, fig_size = None, no_fig = False):
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
        if no_fig:
            return None, None, rows, cols, fig_size
    fig, ax = plt.subplots(rows, cols, figsize=fig_size, squeeze=False)
    ax.reshape(rows, cols)
    for i in range(len(images), rows*cols):
        ax[i//cols, i%cols].axis('off')
    
    if rows == 1 and cols == 1:
        return fig, ax, rows, cols, fig_size
    elif rows == 1:
        ax = ax.reshape(1, cols)
        return fig, ax, rows, cols, fig_size
    elif cols == 1:
        ax = ax.reshape(rows, 1)
        return fig, ax, rows, cols, fig_size
    else:
        return fig, ax, rows, cols, fig_size

def set_legend_location(legend_location, number_of_profiles = 1):
    """
    'upper center', 'lower center', 'upper right', 'upper left', 'lower right', 'lower left', 'center', 'upper', 'lower', 'right', 'left', 'center right', 'center left'
    """
    bbox_to_anchor = bbox_to_anchors[legend_location]
    loc = locs[legend_location]
    
    if legend_location not in ['lower right', 'upper right', 'right', 'center right', 'None']:
        print('Warning: The legend location not known. Please use one of the following: \nlower right, upper right, right, center right, None')
        ncol = number_of_profiles//2 + 2 if number_of_profiles%2 == 0 else number_of_profiles//2 + 3
    else:
        ncol = 1
    
    return bbox_to_anchor, loc, ncol

from matplotlib import patches
def rectangular_frame_for_zoom_boxes(zoom_box, ax, color = 'red', linewidth = 2):
    left, right, buttom, top = zoom_box
    ax.add_patch(patches.Rectangle((left, buttom), right-left, top-buttom, edgecolor = color, facecolor = 'none', linewidth = linewidth))
    return ax

def rectangle_shaper(image, position = 'middle', width = 0.1, height = 0.1, move_h = 0.1, move_v = 0.1):    
    if position == None:
        left = image.shape[1]//4
        buttom = image.shape[0]//4
    if width == None:
        width = 0.1
    if height == None:
        height = 0.1
        
    if position == 'middle':
        left = image.shape[1]//2
        buttom = image.shape[0]//2
    elif position == 'right':
        left = image.shape[1] - width*image.shape[1]
        buttom = image.shape[0]//2 - height*image.shape[0]
    elif position == 'left':
        left = 0
        buttom = image.shape[0]//2 - height*image.shape[0]
    elif position == 'bottom':
        left = image.shape[1]//2
        buttom = image.shape[0] - height*image.shape[0]
    elif position == 'top':
        left = image.shape[1]//2
        buttom = 0
    else:
        #use move_h and move_v to move the rectangle from the middle position
        left = image.shape[1]//2 + move_h*image.shape[1]
        buttom = image.shape[0]//2 + move_v*image.shape[0]
    
    right = width*image.shape[1] + left
    top = height*image.shape[0] + buttom
    return int(left), int(right), int(buttom), int(top)

def _default_zoomboxes():
    return {
        'top right':    [0.7,  0.7, 0.3, 0.3],
        'top left':     [0.0,  0.7, 0.3, 0.3],
        'bottom right': [0.7,  0.0, 0.3, 0.3],
        'bottom left':  [0.0,  0.0, 0.3, 0.3],
        'below':        [0.0, -0.3, 0.3, 0.3],
        'bottom 3':     [[0.0, -0.3, 0.3, 0.3],
                         [0.35, -0.3, 0.3, 0.3],
                         [0.7, -0.3, 0.3, 0.3]],
        'bottom 2':     [[0.0, -0.4, 0.4, 0.4],
                         [0.45, -0.4, 0.4, 0.4]]
    }

def _override_zoomboxes(axin_axis):
    return {
        False: {'obr': [1.1, 0.4, 0.6, 0.6],
                'obl': [-0.5, -0.1, 0.4, 0.6],
                'otr': [1.1,  0.7, 0.3, 0.3]},
        True:  {'obr': [0.7, -0.5, 0.3, 0.3],
                'obl': [0.0, -0.5, 0.3, 0.3],
                'otr': [0.0,  1.1, 0.3, 0.3]}
    }[axin_axis]

def _plot_locations(axin_axis):
    # default vs. alternate for single‐box plots
    single = {
        'obr': [1.1, 0.0, 0.6, 0.4],
        'obl': [-0.5, 0.0, 0.4, 0.4],
        'otr': [1.1, 0.7, 0.3, 0.3],
    }
    alt = {
        'obr': [0.0, -0.5, 0.3, 0.3],
        'obl': [0.7, -0.5, 0.3, 0.3],
        'otr': [0.0,  1.1, 0.3, 0.3],
    }
    return single if not axin_axis else alt

def apply_kwargs(kwargs):    
    global legend_location, colorbar_normalize, colorbar_axins, colorbar_width, colorbar_height, sa_left, sa_right, sa_top, sa_bottom, sa_wspace, sa_hspace, colorbar_size_factor, colorbar_location, shrink, pad, spacing, lw, move_h, move_v, insert_axes, axin_axis, legend_size, use_line_style
    
    legend_location = kwargs.get('legend_location', 'None')
    colorbar_normalize = kwargs.get('colorbar_normalize', False)
    colorbar_axins = kwargs.get('colorbar_axins', None)
    colorbar_width = kwargs.get('colorbar_width', '50%')
    colorbar_height = kwargs.get('colorbar_height', '5%')
    sa_left = kwargs.get('sa_left', 0.1)
    sa_right = kwargs.get('sa_right', 0.9)
    sa_top = kwargs.get('sa_top', 0.9)
    sa_bottom = kwargs.get('sa_bottom', 0.1)
    sa_wspace = kwargs.get('sa_wspace', 0.01)
    sa_hspace = kwargs.get('sa_hspace', 0.2)
    colorbar_size_factor = kwargs.get('colorbar_size_factor', 100)
    colorbar_location = kwargs.get('colorbar_location', 'bottom')
    shrink = kwargs.get('shrink', 0.5)
    pad = kwargs.get('pad', -0.01)
    spacing = kwargs.get('spacing', 'proportional')
    lw = kwargs.get('lw', 4)
    move_h = kwargs.get('move_h', 0)
    move_v = kwargs.get('move_v', 0)
    insert_axes = kwargs.get('insert_axes', True)
    axin_axis = kwargs.get('axin_axis', True)
    legend_size = kwargs.get('legend_size', 20)
    use_line_style = kwargs.get('use_line_style', True)

def apply(kwargs, config = 'libraries/config.json'):
    """
    Visualize images with a flexible JSON-based configuration.

    Parameters:
        images: list or array of images to display.
        config (str or dict, optional): Path to a JSON file or a dict of configuration options.
        **kwargs: Individual parameters to override any values from the config.
    """
    # 1. Load JSON config if provided
    
    if config is not None:
        if isinstance(config, str) and config.lower().endswith('.json'):
            with open(config, 'r') as f:
                cfg = json.load(f) or {}
        elif isinstance(config, dict):
            cfg = config.copy()
        else:
            raise ValueError("`config` must be a JSON filepath or dict.")
        # Merge: explicit kwargs override JSON values
        cfg.update(kwargs)
        params = cfg
    else:
        params = kwargs
    return params

def resample_profile(profile, target_length):
    """
    Resamples the input profile to a target length using linear interpolation.
    
    Parameters:
      profile       : 1D array representing the profile.
      target_length : Desired number of points.
      
    Returns:
      The resampled profile as a 1D array.
    """

    from scipy.interpolate import interp1d
    current_length = len(profile)
    if current_length == target_length:
        return profile
    # Create normalized x-axes for current and target lengths.
    x_current = np.linspace(0, 1, current_length)
    x_target = np.linspace(0, 1, target_length)
    interp_func = interp1d(x_current, profile, kind='linear')
    return interp_func(x_target)
  
def visualize(images, idx = None, rows = None, cols = None, vmode = 'show', cmap = 'coolwarm', title = '', axis = 'on', plot_axis = 'half', min_max = True, dict = None, save_path=None, save_name=None, show_all = False, images_per_row = 5, fig_size = None, coordinates = [None, None, None, None], plot_color = 'blue', color = 'red', colorbar = False, add_to = None, added_image = None, position = None, width = None, height = None, zoomout_location = 'top right', legend_location = 'upper center', overall_title = None, pyqt = False, use_sns = False, use_plotly = False, **kwargs):
    """
    cmaps: Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r',
    legend_locations = ['best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center']
    zoomout_location = ['top right', 'top left', 'bottom right', 'bottom left']
    positions = ['middle', 'random', 'top right', 'top left', 'bottom right', 'bottom left', 'custom', 'middle left', 'middle right', 'middle top', 'middle bottom']
    
    """
    images = convert_images(images, idx)
    title = 'no_title' if title == 'None' or title == 'no' or title == False else title
    if title != 'no_title':
        titles = give_titles(images, title, min_max)
    images = [im[0] if type(im) is list else im for im in images]
    shape = images[0].shape
    dim1 = True
    for i in range(len(shape)-1):
        if shape[i] > 1: 
            dim1 = False
    if dim1:
        #if it's a list of lists, then it's a list of 1D arrays, change to one list using combination
        [plt.plot(images[i]) for i in range(len(images))]
        plt.legend(titles)
        plt.show()
        return None

    if pyqt:
        images = np.stack(images)
        import pyqtgraph as pg
        pg.show(images)
        return None
   
    if dict is not None:
        description_title, add_length = get_setup_info(dict)
    else:
        add_length = None
        
    insert_axes = kwargs['insert_axes'] if 'insert_axes' in kwargs.keys() else True
    axin_axis = kwargs['axin_axis'] if 'axin_axis' in kwargs.keys() else True
    legend_size = kwargs['legend_size'] if 'legend_size' in kwargs.keys() else 20
    colorbar_normalize = kwargs['colorbar_normalize'] if 'colorbar_normalize' in kwargs.keys() else False
    line_styles = ['solid', 'dashed', 'dashdot', 'dotted']
    # colors = ['blue', 'red', 'green', 'orange', 'purple', 'pink', 'brown', 'cyan', 'magenta', 'grey', 'lime', 'teal', 'black','maroon', 'navy', 'olive', 'silver', 'aqua', 'fuchsia', 'lime', 'teal', 'lavender', 'maroon', 'navy', 'olive', 'silver', 'aqua', 'fuchsia', 'blue', 'red', 'green', 'orange', 'purple', 'pink', 'brown', 'cyan', 'magenta', 'grey', 'lime', 'teal', 'black','maroon', 'navy', 'olive', 'silver', 'aqua', 'fuchsia', 'lime', 'teal', 'lavender', 'maroon', 'navy', 'olive', 'silver', 'aqua', 'fuchsia', 'blue', 'red', 'green', 'orange', 'purple', 'pink', 'brown', 'cyan', 'magenta', 'grey', 'lime', 'teal', 'black','maroon', 'navy', 'olive', 'silver', 'aqua', 'fuchsia', 'lime', 'teal', 'lavender', 'maroon', 'navy', 'olive', 'silver', 'aqua', 'fuchsia'] if 'colors' not in kwargs.keys() else kwargs['colors']
    colors = strong_colors

    if type(fig_size) is int:
        fig_size = (fig_size, fig_size)
    
    fig, ax, rows, cols, fig_size= chose_fig(images, rows, cols, kwargs.get('show_all', True), add_length, images_per_row, fig_size)
    upper_limit = rows*cols
    if rows*cols > len(images):
        upper_limit = len(images)
    
    legend_location = kwargs['legend_location'] if 'legend_location' in kwargs.keys() else 'None'
    bbox_to_anchor, loc, ncol = set_legend_location(legend_location, cols*rows)
    apply_kwargs(kwargs)
    cmap = [cmap] * len(images) if type(cmap) != list else cmap
    alpha = [1] * len(images) if 'alpha' not in kwargs.keys() else kwargs['alpha']

    positions = kwargs.get('positions', None)
    zoom_box = None
    
    zoombox_locations = {'top right': [0.7, 0.7, 0.3, 0.3], 'top left': [0.0, 0.7, 0.3, 0.3], 'bottom right': [0.7, 0.0, 0.3, 0.3], 'bottom left': [0.0, 0.0, 0.3, 0.3], 'below': [0.0, -0.3, 0.3, 0.3], 'bottom 3': [[0.0, -0.3, 0.3, 0.3], [0.35, -0.3, 0.3, 0.3], [0.7, -0.3, 0.3, 0.3]], 'bottom 2': [[0.0, -0.4, 0.4, 0.4], [0.45, -0.4, 0.4, 0.4]]}
    zoombox_locations.update({'obr': [1.1, 0.4, 0.6, 0.6], 'obl': [-0.5, -0.1, 0.4, 0.6], 'otr': [1.1, 0.7, 0.3, 0.3]}) if axin_axis == False else zoombox_locations.update({'obr': [0.7, -0.5, 0.3, 0.3], 'obl': [0.0, -0.5, 0.3, 0.3], 'otr': [0.0, 1.1, 0.3, 0.3]})
    zoom_box = zoombox_locations[zoomout_location]
    
    if zoomout_location == 'obr': #outside the box but to the right
        plot_location =[ 1.1, 0.0000, 0.6, 0.4] if axin_axis == False else [0.0, -0.5, 0.3, 0.3] 
    elif zoomout_location == 'obl':
        plot_location = [-0.5, 0.0000, 0.4, 0.4] if axin_axis == False else [0.7, -0.5, 0.3, 0.3]
    elif zoomout_location == 'otr':
        plot_location = [1.1, 0.7, 0.3, 0.3] if axin_axis == False else [0.0, 1.1, 0.3, 0.3]
    elif zoomout_location == 'bottom 3':
        zoom_boxes = [[0.0, -0.3, 0.3, 0.3], [0.35, -0.3, 0.3, 0.3], [0.7, -0.3, 0.3, 0.3]]
        positions = ['custom', 'custom', 'custom']
        plot_location = [0.80, 0.0000, 0.2, 0.2] if axin_axis == False else [0.0, 0.0, 0.3, 0.3]
    elif zoomout_location == 'bottom 2':
        zoom_boxes = [[0.0, -0.4, 0.4, 0.4], [0.45, -0.4, 0.4, 0.4]]
        positions = ['bottom', 'left']
        plot_location =  [0.80, 0.0000, 0.2, 0.2] if axin_axis == False else [0.0, 0.0, 0.3, 0.3]
    else:
        plot_location = [0.00, 0.0000, 0.2, 0.2] if axin_axis == False else [0.0, 0.0, 0.3, 0.3]
    #get the rectangle coordinates
    if coordinates != [None, None, None, None]:
        left, right, buttom, top = coordinates
    else:
        if positions is None: 
            left, right, buttom, top = [], [], [], []
            for image in images:
                    l, r, b, t = rectangle_shaper(image, position = position, width = width, height = height, move_h = move_h, move_v = move_v)
                    left.append(l)
                    right.append(r)
                    buttom.append(b)
                    top.append(t)
        else:
            lefts, rights, buttoms, tops = [], [], [], []
            rects = []
            move_hs = [-0.2, 0.0, 0.2] if 'move_hs' not in kwargs.keys() else kwargs['move_hs']
            move_vs = [-0.2, 0.0, 0.2] if 'move_vs' not in kwargs.keys() else kwargs['move_vs']
            for i in range(len(positions)):
                rect_k = []
                left, right, buttom, top = [], [], [], []
                for j, image in enumerate(images):
                    l, r, b, t = rectangle_shaper(image, position = positions[i], width = width, height = height, move_h = move_hs[i], move_v = move_vs[i]) 
                    # rects.append([l, r, b, t])
                    #convert rects to the range of 0 to 1 with possibility to go beyond 1 and less that 0
                    rect_k.append([l/image.shape[1], t/image.shape[0], r/image.shape[1] - l/image.shape[1], t/image.shape[0] -  b/image.shape[0]])
                    left.append(l)
                    right.append(r)
                    buttom.append(b)
                    top.append(t)
                lefts.append(left)
                rights.append(right)
                buttoms.append(buttom)
                tops.append(top)
                rects.append(rect_k)
            #convert rects to the range of 0 to 1 with possibility to go beyond 1 and less that 0
            
    def show_with_zoom():
        if zoomout_location != 'bottom 3' and zoomout_location != 'bottom 2':                
                if use_sns:
                    import seaborn as sns
                    [sns.heatmap(images[i*cols + j], cmap=cmap[i*cols + j], ax=ax[i,j], cbar=False, cbar_kws={'shrink': shrink, 'pad': pad, 'spacing': spacing})for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
                elif use_plotly:
                    import plotly.express as px
                    [px.imshow(images[i*cols + j], color_continuous_scale=cmap[i*cols + j]) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
                else:
                    [ax[i, j].add_patch(plt.Rectangle((left[i*cols + j], buttom[i*cols + j]), right[i*cols + j] - left[i*cols + j], top[i*cols + j] - buttom[i*cols + j], edgecolor=colors[(i*cols + j)%len(colors)], lw=lw, facecolor='none')) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]  
                
                [ax[i,j].add_patch(plt.Rectangle((left[i*cols + j], buttom[i*cols + j]), right[i*cols + j] - left[i*cols + j], top[i*cols + j] - buttom[i*cols + j], edgecolor=colors[(i*cols+j)%len(colors)], lw=lw, facecolor='none')) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
                axins = [ax[i, j].inset_axes(zoom_box,) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
                [axins[i*cols + j].imshow(images[i*cols + j][buttom[i*cols + j]:top[i*cols + j], left[i*cols + j]:right[i*cols + j]], cmap = cmap[i*cols + j], extent =  [left[i*cols + j], right[i*cols + j], buttom[i*cols+j],  top[i*cols + j]]) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
                [axins[i*cols + j].axis('off') for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
                [axins[i*cols + j].add_patch(plt.Rectangle((left[i*cols + j], buttom[i*cols + j]), right[i*cols + j] - left[i*cols + j], top[i*cols + j] - buttom[i*cols + j], edgecolor=colors[(i*cols+j)%len(colors)], lw=lw, facecolor='none')) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
                # [ax[i,j].indicate_inset_zoom(axins[i*cols + j], edgecolor=colors[(i*cols+j)%len(colors)], alpha = 0.9, lw=5) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
                        
        else:
            
            for k in range(len(positions)):
                axins = []
                for i in range(rows):
                    for j in range(cols):
                        ax[i, j].add_patch(plt.Rectangle((lefts[k][i*cols + j], buttoms[k][i*cols + j]), rights[k][i*cols + j] - lefts[k][i*cols + j], tops[k][i*cols + j] - buttoms[k][i*cols + j], edgecolor=colors[k], lw=lw, facecolor='none'))
                        axin = ax[i, j].inset_axes(zoom_boxes[k], transform=ax[i, j].transAxes) 
                        axins.append(axin)
                [axins[i*cols + j].add_patch(plt.Rectangle((lefts[k][i*cols + j], buttoms[k][i*cols + j]), rights[k][i*cols + j] - lefts[k][i*cols + j], tops[k][i*cols + j] - buttoms[k][i*cols + j], edgecolor=colors[k], lw=lw, facecolor='none')) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
                [axins[i*cols + j].imshow(images[i*cols + j][buttoms[k][i*cols + j]:tops[k][i*cols + j], lefts[k][i*cols + j]:rights[k][i*cols + j]], cmap = cmap[i*cols + j], extent = [lefts[k][i*cols + j], rights[k][i*cols + j],  tops[k][i*cols + j], buttoms[k][i*cols+j], top[i*cols + j]]) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
                [axins[i*cols + j].axis('off') for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
                
                # [ax[i,j].indicate_inset_zoom(axins[i*cols + j], edgecolor=colors[k]) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]   
        return ax, axins
                    
    if colorbar_normalize:
        from matplotlib.colors import Normalize
        from matplotlib import cm
        colorbar_size = 0.07 * (shape[0] / shape[1])
        colorbar_fontsize = colorbar_size * colorbar_size_factor

        # one colorbar for all the images
        min_images = [np.min(images[i*cols + j]) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
        max_images = [np.max(images[i*cols + j]) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
        min_images = np.min(min_images)
        max_images = np.max(max_images)
        norm = Normalize(vmin=min_images, vmax=max_images)
    
    else:
        norm = None

    if use_sns:
        import seaborn as sns
        [sns.heatmap(images[i*cols + j], cmap=cmap[i*cols + j], ax=ax[i,j], cbar=False, cbar_kws={'shrink': shrink, 'pad': pad, 'spacing': spacing})for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
        [ax[i, j].axis(axis) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
    elif use_plotly:
        import plotly.graph_objects as go
        # Create a figure for each image using go.Image and set the colormodel property instead of colorscale.
        
        [go.Figure(go.Image(z=images[i*cols + j], colormodel='rgb'))
        for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]

    else:
        [ax[i,j].imshow(images[i*cols + j], cmap = cmap[i*cols + j], alpha = alpha[i*cols+j], norm = norm) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
        [ax[i, j].axis(axis) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]

    if vmode == 'plot':
        # Adjust subplots for image display
        fig.subplots_adjust(wspace=sa_wspace, hspace=sa_hspace,
                            left=sa_left, right=sa_right, top=sa_top, bottom=sa_bottom)
        
        # Add a rectangle patch on each image (using each image's own shape)
        for idx in range(upper_limit):
            i = idx // cols
            j = idx % cols
            img_shape = images[idx].shape
            ax[i, j].add_patch(plt.Rectangle((1, img_shape[0] // 2),
                                            img_shape[1] - 1, 1,
                                            edgecolor=colors[idx], lw=lw, facecolor='none'))
        
        # Extract profiles from each image
        profiles = []
        for idx in range(upper_limit):
            img = images[idx]
            img_shape = img.shape
            if plot_axis == 'diagonal':
                prof = np.diag(img)
            else:
                prof = img[img_shape[0] // 2, 1:img_shape[1]]
            profiles.append(prof)
        
        # Check if normalization (resampling) is needed by comparing profile lengths.
        unique_lengths = {len(p) for p in profiles}
        if len(unique_lengths) > 1:
            # If sizes differ, choose a common number of points (using the minimum length)
            common_points = min(unique_lengths)
            normalized_profiles = [resample_profile(p, common_points) for p in profiles]
            common_x = np.linspace(0, 1, common_points)
        else:
            normalized_profiles = profiles
            common_x = np.linspace(0, 1, len(profiles[0]))
        
        # Determine line thickness (can be set via kwargs, defaults to 2)
        line_thickness = kwargs.get('line_thickness', 4)
        
        # Plot all the (possibly normalized) profiles in a single figure with different line styles.
        fig2, ax2 = plt.subplots(1, 1, figsize=(20, 10))
        legends = (['im' + str(i + 1) for i in range(len(normalized_profiles))]
                if title == '' or title == 'no_title' else titles)
        for i, prof in enumerate(normalized_profiles):
            
            dash_style = line_styles[i % len(line_styles)]
            ax2.plot(common_x, prof, color=colors[i], label=legends[i], linestyle=dash_style, linewidth=line_thickness) if use_line_style else ax2.plot(common_x, prof, color=colors[i], label=legends[i], linewidth=line_thickness)
              
        # Determine label size based on the smallest image height among all images
        min_img_height = min(img.shape[0] for img in images)
        label_size = kwargs.get('label_size', 25 / (512 / min_img_height))
        ax2.tick_params(axis='both', which='major', labelsize=label_size)
        
        fig2.subplots_adjust(wspace=sa_wspace, hspace=sa_hspace,
                            left=sa_left, right=sa_right, top=sa_top, bottom=sa_bottom)
        if legend_location != 'None':
            ax2.legend(fontsize=label_size, loc=loc, bbox_to_anchor=bbox_to_anchor, ncol=ncol)
        else:
            ax2.legend(fontsize=label_size, loc='right', bbox_to_anchor=(1.0, 0.2),
                    ncol=ncol, fancybox=True, shadow=True)
                    # Remove top and right spines
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        # Only show ticks on the left and bottom
        ax2.xaxis.set_ticks_position('bottom')
        ax2.yaxis.set_ticks_position('left')
        if overall_title is not None:
            ax2.set_title(overall_title, fontsize=label_size * 3 / 2)
        ax2.tick_params(axis='both', which='major', labelsize=label_size * 3 / 2)

                
    elif vmode == 'both':
        [plot_func(ax[i, j], plot_axis, images[i*cols + j], plot_color = colors[(i*cols + j)%len(colors)], add_patch = True, insert_axes = insert_axes, cmap = cmap, axis = axis, axin_axis=axin_axis, rectangle=plot_location) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]

    elif vmode == 'zoom':
        ax, axins = show_with_zoom()

    elif vmode == 'zoom_with_plot':
        ax, axins = show_with_zoom()

        plot_profiles = [np.diag(images[i*cols + j]) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit] if plot_axis == 'diagonal' else [images[i*cols + j][shape[0]//2, 1:shape[1]] for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
        #plot all the profiles in the same image
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 10))
        legends = ['im'+str(i+1) for i in range(len(plot_profiles))] if title == '' else titles
        [ax2.plot(np.arange(1, shape[1]), plot_profiles[i], color=colors[i], label=legends[i]) for i in range(len(plot_profiles))]
        #increase the size of the x-axis and y-axis and formula for the lable size
        label_size = 30/(1024/shape[0])
        ax2.tick_params(axis='both', which='major', labelsize=label_size)
        #add overall title to the plot 
        ax2.legend(fontsize=label_size, loc=loc, bbox_to_anchor=bbox_to_anchor, ncol = ncol) if legend_location != 'None' else ax2.legend(fontsize=label_size)
        if overall_title is not None:
            ax2.set_title(overall_title, fontsize = label_size*2)
        else:
            ax2.set_title('Profiles', fontsize =  label_size*2)

    elif vmode == 'zoom_with_plot_line':
        ax, axins = show_with_zoom()
        [ax[i, j].add_patch(plt.Rectangle((1, shape[0]//2), shape[1] - 1, 1, edgecolor=colors[(i*cols + j)%len(colors)], lw=lw, facecolor='none')) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]

    elif vmode == 'all':
        ax, axins = show_with_zoom()
        [plot_func(ax[i, j], plot_axis, images[i*cols + j], plot_color = colors[(i*cols + j)%len(colors)], add_patch = True, insert_axes = insert_axes, cmap = cmap, axis = axis, axin_axis=axin_axis, rectangle=plot_location) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
    
    elif vmode == 'add' or vmode == 'add_show' or vmode ==  'add_plot':
        assert added_image is not None, "added_image is None"
        if 'axes_given' in kwargs.keys():
            axes_given = kwargs['axes_given']
        else:
            axes_given = [0, 0.8, 0.2, 0.2]
        if type(added_image) is not list:
            added_image = [added_image] * len(images)
        if add_to is None:
            axins = [ax[i,j].inset_axes([0.1, 0.1, 0.3, 0.3]) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
            [axins[i*cols + j].imshow(added_image[i*cols + j], cmap = cmap[i*cols + j], alpha = alpha[i*cols+j]) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
            [axins[i*cols + j].axis('off') for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
        
        else:
            if type(add_to) == tuple:
                add_to_i = add_to[0]
                add_to_j = add_to[1]
            elif type(add_to) == int:
                add_to_i = add_to//cols
                add_to_j = add_to%cols
            #add a third image to the selected ones
            for i in range(rows):
                for j in range(cols):
                    if i == add_to_i and j == add_to_j:
                        ax[i,j].imshow(images[i*cols + j], cmap = cmap[i*cols + j])
                        axins = ax[i,j].inset_axes(axes_given)
                        axins.imshow(added_image[i*cols + j],cmap = cmap[i*cols + j])
                        axins.axis('off')
                    else:
                        ax[i,j].imshow(images[i*cols + j], cmap = cmap[i*cols + j])
                        axins.axis('off')
    

        if vmode == 'add_plot':
            if plot_axis == 'half':
                axins2 = [ax[i,j].inset_axes(plot_location) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
                [axins2[i*cols + j].plot(np.arange(1, images[i*cols + j].shape[1]), images[i*cols + j][images[i*cols + j].shape[0]//2, :], color=plot_color) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
                [axins2[i*cols + j].axis('off') for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]

            else:
                assert type(plot_axis) == int or type(plot_axis) == float, "plot_axis should be an integer or a float"
                axins2 = [ax[i,j].inset_axes(plot_location) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
                [axins2[i*cols + j].plot(np.arange(1, images[i*cols + j].shape[1]), images[i*cols + j][plot_axis, :], color=plot_color) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
                [axins2[i*cols + j].axis('off') for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]


        elif vmode == 'add_all':
            [ax[i,j].add_patch(plt.Rectangle((left[i*cols + j], buttom[i*cols + j]), right[i*cols + j] - left[i*cols + j], top[i*cols + j] - buttom[i*cols + j], edgecolor=colors[(i*cols + j)%len(colors)], lw=lw, facecolor='none')) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
            [ax[i,j].add_patch(plt.Rectangle((1, buttom[i*cols + j]+(top[i*cols + j]-buttom[i*cols + j])//2), right[i*cols + j] - 1, 1, edgecolor=plot_color, lw=lw, facecolor='none')) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
            axins = [ax[i,j].inset_axes([0.7, 0.7, 0.3, 0.3]) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
            [axins[i*cols + j].imshow(images[i*cols + j][buttom[i*cols + j]:top[i*cols + j], left[i*cols + j]:right[i*cols + j]], cmap = cmap) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
            [axins[i*cols + j].axis('off') for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
            axins2 = [ax[i,j].inset_axes([0.00, 0.0000, 0.3, 0.3]) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
            [axins2[i*cols + j].plot(np.arange(1, right[i*cols + j]), images[i*cols + j][buttom[i*cols + j]+(top[i*cols + j]-buttom[i*cols + j])//2, 1:right[i*cols + j]], color=plot_color) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
            [axins2[i*cols + j].axis('off') for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
                

    if title != 'no_title':
        fontsize = kwargs['fontsize'] if 'fontsize' in kwargs.keys() else 19
        # [ax[i, j].set_title(titles[i*cols + j], fontsize=fontsize, y = 1.0, pad=-14) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
        #pad the title
        title_x = kwargs['title_x'] if 'title_x' in kwargs.keys() else 0.5
        title_y = kwargs['title_y'] if 'title_y' in kwargs.keys() else 1
        title_color = kwargs['title_color'] if 'title_color' in kwargs.keys() else 'black'
        title_color = [title_color] * len(images) if type(title_color) != list else title_color
        title_horizontalalignment = kwargs['title_horizontalalignment'] if 'title_horizontalalignment' in kwargs.keys() else 'center'
        title_fontweight = kwargs['title_fontweight'] if 'title_fontweight' in kwargs.keys() else 'bold'
        [ax[i, j].set_title(titles[i*cols + j], fontsize=fontsize, x = title_x, y = title_y, color = title_color[(i*cols+j)%len(title_color)], horizontalalignment=title_horizontalalignment, fontweight = title_fontweight) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
        
    #if there is a secoond title
    second_title = kwargs['second_title'] if 'second_title' in kwargs.keys() else 'no_title'
    #add it as a text
    if second_title != 'no_title':
        #color white, type bold
        second_title_x = kwargs['second_title_x'] if 'second_title_x' in kwargs.keys() else 0.3
        second_title_y = kwargs['second_title_y'] if 'second_title_y' in kwargs.keys() else 0.05
        second_title_color = kwargs['second_title_color'] if 'second_title_color' in kwargs.keys() else 'w'
        second_title_fontsize = kwargs['second_title_fontsize'] if 'second_title_fontsize' in kwargs.keys() else 28
        second_title_horizontalalignment= kwargs['second_title_horizontalalignment'] if 'second_title_horizontalalignment' in kwargs.keys() else 'center'
        second_title_fontweight = kwargs['second_title_fontweight'] if 'second_title_fontweight' in kwargs.keys() else 'bold'
        if type(second_title_color) != list:
            second_title_color = [second_title_color] * len(images)
        
        [ax[i, j].text(second_title_x, second_title_y, second_title[i*cols + j], horizontalalignment=second_title_horizontalalignment, verticalalignment='center', transform=ax[i, j].transAxes,  c= second_title_color[(i*cols+j)%len(second_title_color)],fontweight = second_title_fontweight,fontsize=second_title_fontsize) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
        
        
    # plt.tight_layout()
    if vmode != 'plot':
        fig.patch.set_facecolor('xkcd:white')
    if colorbar_normalize:
        colorbar = False
    if colorbar:
        if type(colorbar) is not list:
            colorbar = [colorbar] * len(images)
        if type(cmap) is not list:
            cmap = [cmap] * len(images)
        colorbar_size = 0.07 * (shape[0] / shape[1])
        colorbar_fontsize = colorbar_size * colorbar_size_factor
        
        if colorbar_axins != None:
            cols = [fig.colorbar(ax[i, j].imshow(images[i*cols + j], cmap = cmap[i*cols + j]), ax=ax[i, j], cax = insert_axins(ax[i, j],colorbar_axins, colorbar_width, colorbar_height), location = colorbar_location, spacing = spacing, fraction = colorbar_size, shrink = shrink, pad = pad) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
        else:
            cols = [fig.colorbar(ax[i, j].imshow(images[i*cols + j], cmap = cmap[i*cols + j]), ax=ax[i, j], location = colorbar_location, spacing = spacing, fraction = colorbar_size, shrink = shrink, pad = pad) for i in range(rows) for j in range(cols) if i*cols + j < upper_limit]
        #adjust the font
        [cols[i].ax.xaxis.set_tick_params('major', labelsize=colorbar_fontsize*5.5, pad = 0, rotation=0) for i in range(len(cols))]
        [cols[i].ax.yaxis.set_tick_params('major', labelsize=colorbar_fontsize*5.5) for i in range(len(cols))]
        #adjust the colorbar
        # [cols[r].set_label('$\it{(min, max):}$('+str(np.min(images[r]))+' , '+str(np.max(images[r]))+')', fontsize=fontsize) for r in range(len(cols))]
    
    if dict is not None:
        fig.subplots_adjust(left=add_length/150)
        fig.suptitle(description_title, fontsize=10, y=0.95, x=0.05, ha='left', va='center', wrap=True, color='blue')

    if save_path is not None:
        if save_name is None:
            save_name = get_file_nem(dict) if dict is not None else 'image'
        save_path = save_path + save_name + '.png'
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=300, facecolor='w', edgecolor='w', orientation='portrait', transparent=False, bbox_extra_artists=None, metadata=None)
    

    plt.subplots_adjust(left=sa_left, bottom=sa_bottom, right=sa_right, top=sa_top, wspace=sa_wspace, hspace=sa_hspace)
    
    if colorbar_normalize:            
        plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap[0]), ax=ax, shrink = shrink, pad = pad, location = colorbar_location, fraction = colorbar_size).ax.tick_params(labelsize=colorbar_fontsize*5.5)
        
    plt.show()        
    # return fig, ax, plt

class IV:
    """ImageVisualizer class for plotting images."""
    def __init__(self, images = None, config = 'libraries/config.json', **kwargs):
        params = apply(kwargs, config)
        for key, value in params.items():
            setattr(self, key, value)
        self._prepare_images(images, self.idx)
        self.title = self.get_title()
        self.use_pyqt if self.pyqt else None
        description_title, add_length = get_setup_info(self.dict) if self.dict is not None else ('', None)
        with open('libraries/strong_colors.json', 'r') as f:
            self.strong_colors = json.load(f)
        with open('libraries/zoombox_locations.json', 'r') as f:
            self.zoombox_locations = json.load(f)
        if self.axin_axis:
            with open('libraries/plot_locations_axin_axis.json', 'r') as f:
                self.plot_locations = json.load(f)
        else:
            with open('libraries/plot_locations_no_axin_axis.json', 'r') as f:
                self.plot_locations = json.load(f)
        self.zoom_box = self.zoombox_locations[self.zoombox_location] 
        print(f"Using zoombox location: {self.zoombox_location}, coordinates: {self.zoom_box}") 
        self.plot_locations = self.plot_locations[self.zoombox_location]
        self.coordinates = self.get_coordinate()
        

        
    def _prepare_images(self, images = None, idx = None):
        """Prepare images for visualization."""
        images = self.images if images == None else images    
        images = convert_images(images, idx)
        self.images = [im[0] if type(im) is list else im for im in images]
        self.shapes = [im[0].shape for im in self.images]
        self.fig, self.ax, self.rows, self.cols, fig_size= chose_fig(self.images, self.idx, self.rows, self.cols, self.add_length, self.show_all, self.images_per_row, self.fig_size)
        self.upper_limit = len(images) if  self.rows* self.cols > len(self.images) else  self.rows* self.cols
        self.cmap = [self.cmap] * len(self.images) if type(self.cmap) is not list else self.cmap
        if len(self.cmap) < len(self.images):
            self.cmap += [self.cmap[-1]] * (len(self.images) - len(self.cmap))
        self.alpha = [self.alpha] * len(self.images)
        self.fig_size = (self.fig_size, self.fig_size) if type(self.fig_size) is int else self.fig_size
        
        
    
    def get_title(self):
        """Get the title for the image at index idx."""
        title = self.title
        title = 'no_title' if title == None or title == 'no' or title == False else title
        if title != 'no_title':
            titles = give_titles(self.images, title, self.min_max)
            return titles
        else:
            return 'no_title'
    
    def use_pyqt(self):
        """Check if PyQt is used for visualization."""
        import pyqtgraph as pg
        if not all(im.shape == self.images[0].shape for im in self.images):
            max_shape = max(im.shape for im in self.images)
            images = [resize_with_diff_interpolation(im, (max_shape[1], max_shape[0])) for im in self.images]
        images = np.stack(images)
        pg.show(images)
        return None
     
    def get_coordinate(self):
        left, right, buttom, top = [], [], [], []
        lefts, rights, buttoms, tops = [], [], [], []
        rects = []
        #get the rectangle coordinates
        if self.coordinates != [None, None, None, None]:
            left, right, buttom, top = self.coordinates
            lefts, rights, buttoms, tops = [left], [right], [buttom], [top]
        else:
            if self.positions is None: 
                for image in self.images:
                        l, r, b, t = rectangle_shaper(image, position = self.position, width = self.width, height = self.height, move_h = self.move_h, move_v = self.move_v)
                        left.append(l)
                        right.append(r)
                        buttom.append(b)
                        top.append(t)
                lefts, rights, buttoms, tops = [left], [right], [buttom], [top]
            else:
                for i in range(len(self.positions)):
                    rect_k = []
                    left, right, buttom, top = [], [], [], []
                    for j, image in enumerate(self.images):
                        l, r, b, t = rectangle_shaper(image, position = self.positions[i], width = self.width, height = self.height, move_h = self.move_hs[i], move_v = self.move_vs[i]) 
                        rect_k.append([l/image.shape[1], t/image.shape[0], r/image.shape[1] - l/image.shape[1], t/image.shape[0] -  b/image.shape[0]])
                        left.append(l)
                        right.append(r)
                        buttom.append(b)
                        top.append(t)
                    lefts.append(left)
                    rights.append(right)
                    buttoms.append(buttom)
                    tops.append(top)
                    rects.append(rect_k)
        self.lefts, self.rights, self.buttoms, self.tops, self.rects = lefts, rights, buttoms, tops, rects
        return lefts, rights, buttoms, tops, rects        
    
    
    