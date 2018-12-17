import texfig
import matplotlib.pyplot as plt
import numpy as np
import os
import configparser
from os import path
import pandas as pd
import matplotlib.gridspec as gridspec
import io_util as iot
import locale
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, DrawingArea, HPacker
from cycler import cycler
import matplotlib.path as mpath
import mri_draw_utils as mrd
import math_format as mf
import matplotlib.ticker as ticker


config_loc = path.join('config')
config_filename = 'result.config'
config_file = os.path.join(config_loc, config_filename)
config = configparser.ConfigParser()
config.read(config_file)
config = config

roi_volume_folder = config.get('results', 'roi_volume_folder')
structural_figures = config.get('results', 'structural_figures')

data2D_path = 'tsc_cost_combined_2d.csv'
data3D_path = 'tsc_cost_combined_3d.csv'
data3D_path = 'tsc_cost_combined_4d.csv'
data_all_path = 'tsc_cost_combined_all.csv'
data_all_by_volume = 'tsc_cost_combined_all_by_volumev1.csv'
data_all_by_volume2 = 'tsc_cost_combined_all_by_volumev2.csv'
data_by_volume2D = 'tsc_cost_combined_by_volume2d.csv'
data_by_volume3D = 'tsc_cost_combined_by_volume3d.csv'
data_by_volume4D = 'tsc_cost_combined_by_volume4d.csv'
data_by_mr_by_volume3D = 'tsc_cost_combined_by_mr_volume3d.csv'
data_by_mr_by_volume2D = 'tsc_cost_combined_by_mr_volume2d.csv'
data_by_mr_by_volume4D = 'tsc_cost_combined_by_mr_volume4d.csv'

top_legend_handles = []
top_labels = []

top_legend_handles3D = []
top_labels3D = []

top_legend_handles2D = []
top_labels2D = []

top_legend_handles3D_mr = []
top_labels3D_mr = []

top_legend_handles4D_mr = []
top_labels4D_mr = []

top_legend_handles2D_mr = []
top_labels2D_mr = []

top_legend_tensor_dim = []
top_labelstensor_dim = []

star = mpath.Path.unit_regular_star(6)
circle = mpath.Path.unit_circle()
# concatenate the circle with an internal cutout of the star
verts = np.concatenate([circle.vertices, star.vertices[::-1, ...]])
codes = np.concatenate([circle.codes, star.codes])
cut_star = mpath.Path(verts, codes)

tex_width = 5.78853 # in inches
tex_width = 4.78853 # in inches

bg_legend_handle = plt.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)

def format_axes(ax):

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color('k')
        ax.spines[spine].set_linewidth(1)

    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color('k')
        ax.spines[spine].set_linewidth(1)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='in', top = True, left = True, color='k')

    return ax

def legend_outside(ncol, extra_text):
    handles, labels = plt.gca().get_legend_handles_labels()
    lgd = plt.legend(handles, labels, loc='upper center', ncol=ncol, bbox_to_anchor=(-0.15, -0.3))
    import matplotlib.offsetbox as offsetbox 
    txt=offsetbox.TextArea(extra_text, textprops=dict(size=lgd.get_texts()[0].get_fontsize()))
    box = lgd._legend_box
    box.get_children().append(txt) 
    box.set_figure(box.figure)
    return lgd

def create_legend_on_top(ncol, handles, labels, extra_text):
    lgd = plt.legend(handles, labels, loc='lower left', ncol=ncol, bbox_to_anchor=(0., 1.02, 1., .102))
    import matplotlib.offsetbox as offsetbox 
    txt=offsetbox.TextArea(extra_text, textprops=dict(size=7))
    box = lgd._legend_box
    box.get_children().append(txt) 
    box.set_figure(box.figure)
    return lgd

def create_legend_on_inside_upper_left(ncol, handles, labels, extra_text):
    lgd = plt.legend(handles, labels, loc='upper left', ncol=ncol, bbox_to_anchor=(0.01, 0.98), fontsize="small", edgecolor='k')
    import matplotlib.offsetbox as offsetbox 
    txt=offsetbox.TextArea(extra_text, textprops=dict(size=7))
    box = lgd._legend_box
    box.get_children().append(txt) 
    box.set_figure(box.figure)
    return lgd

def create_legend_on_inside_upper_left_inset(ncol, handles, labels, extra_text, x, y):
    lgd = plt.legend(handles, labels, loc='upper left', ncol=ncol, bbox_to_anchor=(x, y), fontsize="small", edgecolor='k')
    lgd.set_zorder(20)
    import matplotlib.offsetbox as offsetbox 
    txt=offsetbox.TextArea(extra_text, textprops=dict(size=7))
    box = lgd._legend_box
    box.get_children().append(txt) 
    box.set_figure(box.figure)
    return lgd

def create_legend_on_bottom(ncol, handles, labels, extra_text):
    lgd = plt.legend(handles, labels, loc='upper center', ncol=ncol, bbox_to_anchor=(-0.15, -0.17),fontsize="x-small", edgecolor='k')
    return lgd

def create_legend_on_bottom_inset(ncol, handles, labels, extra_text, x, y):
    lgd = plt.legend(handles, labels, loc='upper center', ncol=ncol, bbox_to_anchor=(x, y),fontsize="x-small", edgecolor='k')
    return lgd

def legend_outside1(ncol, extra_text):
    handles, labels = plt.gca().get_legend_handles_labels()
    lgd = plt.legend(handles, labels, loc='upper center', ncol=ncol, bbox_to_anchor=(0.55, -0.15), frameon=False)
    import matplotlib.offsetbox as offsetbox 
    txt=offsetbox.TextArea(extra_text, textprops=dict(size=7))
    box = lgd._legend_box
    box.get_children().append(txt) 
    box.set_figure(box.figure)
    return lgd

def group(number):
    s = '%d' % number
    groups = []
    while s and s[-1].isdigit():
        groups.append(s[-3:])
        s = s[:-3]
    return s + ','.join(reversed(groups))


def plot_volume(data, metric_name, filter_name,  **kwargs):
    data_size = data.loc[data['roi_volume_label'] == filter_name]
    x = data_size['mr']
    y = data_size[metric_name]
    roi_volume = data_size.tail(1)['el_volume']
    lw = 1
    label = ur'ROI Volume: ' + group(roi_volume)
    result = plt.plot(x, y, linewidth=lw, label = "")
    return label, result

def plot_mr(data, metric_name, filter_name,  **kwargs):
    data_size = data.loc[data['mr'] == filter_name]
    y = data_size[metric_name]
    x = data_size['el_volume']
    lw = 1
    label = r'MR ' + str(filter_name) + "%"
    result = plt.plot(x, y, linewidth=lw, label = "")
    return label, result

def plot_scatter_mr(data, metric_name, filter_name,  marker, color, size = 2, alpha = 1, **kwargs):
    data_size = data.loc[data['mr'] == filter_name]
    x = data_size['el_volume']
    y = data_size[metric_name]
    tr_count = data_size.tail(1)['ts_count']
    filter_name_str = str(filter_name)
    label = ur'MR '  + filter_name_str +  ur'%' 
    return label, plt.scatter(x, y, label = "",  c = color, s = size, marker =  marker, alpha = alpha)


def set_axis_style(ax, labels, name):
    ax.get_xaxis().set_tick_params(direction='in')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel(name)
    

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value
    
def plot_violin_mr(data, metric_name, filter_name,  ax, **kwargs):
    violin_data = []
    
    colors = ["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#AD002AFF", "#ADB6B6FF"]
    labels_x = []
    
    for k in range(5,35, 5):
        data_size = data.loc[data['mr'] == k]
        y = data_size[metric_name]
        violin_data.append(y)
        
    n = 6
    for i in range(n):
        filter_label = 'size' + str(i)
        row = data.loc[data['roi_volume_label'] == filter_label]
        item_label = row['el_volume']
        labels_x.append(item_label)
        
    parts = ax.violinplot(violin_data, vert=True, showmeans=False, showmedians=False,
        showextrema=False)
    
    ctr = 0
    for pc in parts['bodies']:
        pc.set_facecolor(colors[ctr])
        pc.set_edgecolor('black')
        pc.set_lw(1)
        pc.set_alpha(0.8)
        ctr = ctr + 1
    
    violin_data_as_arr = np.array(violin_data)   
    quartile1, medians, quartile3 = np.percentile(violin_data_as_arr, [25, 50, 75], axis=1)
    whiskers = np.array([
    adjacent_values(sorted_array, q1, q3)
    for sorted_array, q1, q3 in zip(violin_data_as_arr, quartile1, quartile3)])
    whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]

    inds = np.arange(1, len(medians) + 1)
    ax.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
    ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    ax.vlines(inds, whiskersMin, whiskersMax, color='k', linestyle='-', lw=1)
    
    axis_label = 'x'
    #set_axis_style(ax, labels_x, axis_label)
    
    return labels_x, parts

def plot_scatter(data, metric_name, filter_name,  marker, color, size = 2, alpha = 1, **kwargs):
    data_size = data.loc[data['roi_volume_label'] == filter_name]
    x = data_size['mr']
    y = data_size[metric_name]
    roi_volume = data_size.tail(1)['el_volume']
    label_size = 'ROI Volume: ' + group(roi_volume)
    return plt.scatter(x, y, label = "",  c = color, s = size, marker =  marker, alpha = alpha)

def draw_4D_roi_volume():
   
    file_path = os.path.join(roi_volume_folder, data_by_volume4D)
    data = iot.read_data_structural(file_path)
    data_size0 = data.loc[data['roi_volume_label'] == 'size0']
    fig = texfig.figure(frameon = False)
    
    print data_size0
   
    title = "Effect of ROI Volume on 4D fMRI Completion by Missing Values Rate \n  Structural Missing Value Pattern";
    fig.suptitle(title, fontsize=8)
    
    grid_rows = 1
    grid_cols = 2
    
    grid = gridspec.GridSpec(grid_rows,grid_cols, hspace=0.05, wspace=0.05)
    
    lw = 2
    tsc_ax = fig.add_subplot(grid[0, 0])
    tsc_ax.set_xlabel("Missing Values %")
    tsc_x = data_size0['mr']
    tsc_y = data_size0['tsc_cost']
    
    roi_volume = data_size0.tail(1)['el_volume']
    label_size0 = 'ROI Volume: ' + group(roi_volume)
    
    print label_size0
    plt.plot(tsc_x, tsc_y, linewidth=lw,
             label= label_size0)
    
    plt.legend(loc='upper right', fontsize=6)
    solution_id = 'structural_by_roi_volume'
    file_path = os.path.join(structural_figures, solution_id)
    texfig.savefig(file_path, pad = 2)
    
    pass

def draw_4D_by_roi_volume():
    file_path = os.path.join(roi_volume_folder, data_by_mr_by_volume4D)
    data = iot.read_data_structural(file_path)

    n = 6
      
    LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
    NUM_STYLES = len(LINE_STYLES)
    
    grid_rows = 1
    grid_cols = 2
    plt.clf()
    fig, (ax0, ax1) = texfig.subplots(width=tex_width, pad = 0, nrows=grid_rows, ncols=grid_cols, sharex=True, sharey=False)
    title = ur'Effect of ROI Size on 4D fMRI Completion by Missing Values Rate  - ' + ur'Structural Missing Value Pattern';
    fig.suptitle(title, fontsize=8, fontweight='semibold')
    fig.subplots_adjust(wspace = 0.3, right = 0.95, bottom = 0.3)
    
    ax0.set_ylabel(ur'Tensor Completion Score')
    ax0.set_xlabel(ur'Missing Values (%)')
    ax0.tick_params(direction="in")
    plt.xticks(range(5, 35, 5), fontsize=8)
    
    ax0.tick_params(direction="in")
    format_axes(ax0)
    
    #colors = ["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#FDAF91FF", "#AD002AFF", "#ADB6B6FF"]
    colors = ["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#AD002AFF", "#ADB6B6FF"]
    ax0.set_prop_cycle(cycler('color', colors))
    ax0.set_ylim(-0.035,0.6)
    
    markers = []
    markers.append("o")
    markers.append("^")
    markers.append("s")
    markers.append(">")
    markers.append("8")
    markers.append("D")

    for i in range(n):
        filter_name = 'size' + str(i)
        metric_name = 'tsc_cost'
        plt.sca(ax0)
        
        label, lines = plot_volume(data, metric_name, filter_name)
        
        lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
        scatter = plot_scatter(data, metric_name, filter_name, markers[i], colors[i], 25, alpha = 0.5)
        
        top_legend_handles.append(lines[0])
        top_labels.append(label)
    
      
    # tcs by z_ score 
    ax1.set_prop_cycle(cycler('color', colors))
    ax1.set_ylabel(ur'TCS $\vert Z_{score} \vert > 2$')
    ax1.set_xlabel(ur'Missing Values (%)')
    ax1.tick_params(direction="in")
    ax1.set_ylim(-0.035,0.6)
    format_axes(ax1)
    
    for k in range(n):
        filter_name = 'size' + str(k)
        metric_name = 'tsc_z_cost'
        plt.sca(ax1)
        
        label, lines = plot_volume(data, metric_name, filter_name)
        lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
        plot_scatter(data, metric_name, filter_name, markers[k], colors[k], 25, alpha = 0.5)
    
    lgd = create_legend_on_bottom(3, top_legend_handles, top_labels, "")
    
    
    tensorDimAnn = TextArea(ur'\textbf{Tensor Dimension}' + ur' = ' + ur'$4$')
    boxUp = HPacker(children=[tensorDimAnn],
              align="center",
              pad=0, sep=5)
    
    tensorDimAnnAnchored_box = AnchoredOffsetbox(loc=8,
                                 child=boxUp, pad=0.,
                                 frameon=False,
                                 bbox_to_anchor=(-0.25, 1.03),
                                 bbox_transform=ax1.transAxes,
                                 borderpad=0.
                                 )
    
    ax1.add_artist(tensorDimAnnAnchored_box)
    
    
    figureA = TextArea(ur'(a) Tensor Completion Score', textprops=dict(color="k", size="x-small"))
    box = HPacker(children=[figureA],
              align="center",
              pad=0, sep=5)
    
    figureAnchored_box = AnchoredOffsetbox(loc=9,
                                 child=box, pad=0.,
                                 frameon=False,
                                 bbox_to_anchor=(0.55, -0.35),
                                 bbox_transform=ax0.transAxes,
                                 borderpad=0.
                                 )
    
    ax0.add_artist(figureAnchored_box)
    
    
    figureB = TextArea(ur'(b) Tensor Completion Score ' + ur'$\vert Z_{score}\vert > 2$', textprops=dict(color="k", size="x-small"))
    box = HPacker(children=[figureB],
              align="center",
              pad=0, sep=5)
    
    figureBAnchored_box = AnchoredOffsetbox(loc=9,
                                 child=box, pad=0.,
                                 frameon=False,
                                 bbox_to_anchor=(0.55, -0.35),
                                 bbox_transform=ax1.transAxes,
                                 borderpad=0.
                                 )
    
    ax1.add_artist(figureBAnchored_box)
    
    solution_id = 'structural_effect_roi_volume_by_mr_4D'
    file_path = os.path.join(structural_figures, solution_id)
    texfig.savefig_pub(file_path, additional_artists=(lgd,)) 
    
def draw_4D_by_roi_volume_and_mr():
    file_path = os.path.join(roi_volume_folder, data_by_volume4D)
    print ("Data Path:" + str(file_path))
    data = iot.read_data_structural(file_path)

    n = 6
     
    LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
    NUM_STYLES = len(LINE_STYLES)
    
    grid_rows = 1
    grid_cols = 2
    plt.clf()
    fig, (ax0, ax1) = texfig.subplots(width=tex_width, pad = 0, nrows=grid_rows, ncols=grid_cols, sharex=True, sharey=False)
    title = ur'Effect of Missing Values Rate on 4D fMRI Completion by ROI Size  - ' + ur'Structural Missing Value Pattern';
    fig.suptitle(title, fontsize=8, fontweight='semibold')
    fig.subplots_adjust(wspace = 0.3, right = 0.95, bottom = 0.3)
    
    ax0.set_ylabel(r'Tensor Completion Score')
    ax0.set_xlabel(r'ROI Size ' + r'$(voxel^3)$')
    
    ax0.tick_params(direction="in")
    
    format_axes(ax0)
    
    x_ticks = []
    n = 6
    for i in range(n):
        filter_label = 'size' + str(i)
        row = data.loc[data['roi_volume_label'] == filter_label]
        item_label = row['el_volume']
        x_ticks.append(item_label)
        
    x_ticks = [2346,2681,3016,3351,3686,4021]
        
    plt.xticks(x_ticks, fontsize=8)
    #colors = ["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#FDAF91FF", "#AD002AFF", "#ADB6B6FF"]
    colors = ["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#AD002AFF", "#ADB6B6FF"]
    ax0.set_prop_cycle(cycler('color', colors))
    ax0.set_ylim(-0.035,0.6)
    
    markers = []
    markers.append("o")
    markers.append("^")
    markers.append("s")
    markers.append(">")
    markers.append("8")
    markers.append("D")

    ctr = 0
    for i in range(5,35, 5):
        filter_name = i
        metric_name = 'tsc_cost'
        plt.sca(ax0)
    
        label, lines = plot_mr(data, metric_name, filter_name)
        lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
        label, scatter = plot_scatter_mr(data, metric_name, filter_name, markers[ctr], colors[ctr], 25, alpha = 0.5)
        
        top_legend_handles4D_mr.append(scatter)
        top_labels4D_mr.append(label)
        ctr = ctr + 1
    
      
    # tcs by z_ score 
    ax1.set_prop_cycle(cycler('color', colors))
    ax1.set_ylabel(ur'TCS $\vert Z_{score} \vert > 2$')
    ax1.set_xlabel(r'ROI Size ' + r'$(voxel^3)$')
    ax1.tick_params(direction="in")
    ax1.set_ylim(-0.035,0.6)
    format_axes(ax1)
    
    ctr = 0
    for k in range(5,35, 5):
        filter_name = k
        metric_name = 'tsc_z_cost'
        plt.sca(ax1)
        
        label, lines = plot_mr(data, metric_name, filter_name)
        lines[0].set_linestyle(LINE_STYLES[k%NUM_STYLES])
        label, scatter = plot_scatter_mr(data, metric_name, filter_name, markers[ctr], colors[ctr], 25, alpha = 0.5)
        ctr = ctr + 1
    
    lgd = create_legend_on_bottom(3, top_legend_handles4D_mr, top_labels4D_mr, "")
    
    
    tensorDimAnn = TextArea(ur'\textbf{Tensor Dimension}' + ur' = ' + ur'$4$')
    boxUp = HPacker(children=[tensorDimAnn],
              align="center",
              pad=0, sep=5)
    
    tensorDimAnnAnchored_box = AnchoredOffsetbox(loc=8,
                                 child=boxUp, pad=0.,
                                 frameon=False,
                                 bbox_to_anchor=(-0.25, 1.03),
                                 bbox_transform=ax1.transAxes,
                                 borderpad=0.
                                 )
    
    ax1.add_artist(tensorDimAnnAnchored_box)
    
    figureA = TextArea(ur'(a) Tensor Completion Score', textprops=dict(color="k", size="x-small"))
    box = HPacker(children=[figureA],
              align="center",
              pad=0, sep=5)
    
    figureAnchored_box = AnchoredOffsetbox(loc=9,
                                 child=box, pad=0.,
                                 frameon=False,
                                 bbox_to_anchor=(0.55, -0.35),
                                 bbox_transform=ax0.transAxes,
                                 borderpad=0.
                                 )
    
    ax0.add_artist(figureAnchored_box)
    
    
    figureB = TextArea(ur'(b) Tensor Completion Score ' + ur'$\vert Z_{score}\vert > 2$', textprops=dict(color="k", size="x-small"))
    box = HPacker(children=[figureB],
              align="center",
              pad=0, sep=5)
    
    figureBAnchored_box = AnchoredOffsetbox(loc=9,
                                 child=box, pad=0.,
                                 frameon=False,
                                 bbox_to_anchor=(0.55, -0.35),
                                 bbox_transform=ax1.transAxes,
                                 borderpad=0.
                                 )
    
    ax1.add_artist(figureBAnchored_box)
    
            
    solution_id = 'structural_effect_mr_by_roi_volume4D'
    file_path = os.path.join(structural_figures, solution_id)
    texfig.savefig_pub(file_path, additional_artists=(lgd,))
    

def draw_3D_by_roi_volume_and_mr():
    file_path = os.path.join(roi_volume_folder, data_by_volume2D)
    print ("Data Path:" + str(file_path))
    data = iot.read_data_structural(file_path)

    n = 6
     
    LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
    NUM_STYLES = len(LINE_STYLES)
    
    grid_rows = 1
    grid_cols = 2
    plt.clf()
    fig, (ax0, ax1) = texfig.subplots(width=tex_width, pad = 0, nrows=grid_rows, ncols=grid_cols, sharex=True, sharey=False)
    title = ur'Effect of Missing Values Rate on 4D fMRI Completion by ROI Size  - ' + ur'Structural Missing Value Pattern';
    fig.suptitle(title, fontsize=8, fontweight='semibold')
    fig.subplots_adjust(wspace = 0.3, right = 0.95, bottom = 0.3)
    
    ax0.set_ylabel(r'Tensor Completion Score')
    ax0.set_xlabel(r'ROI Size ' + r'$(voxel^3)$')
    
    ax0.tick_params(direction="in")
    
    format_axes(ax0)
    
    x_ticks = []
    n = 6
    for i in range(n):
        filter_label = 'size' + str(i)
        row = data.loc[data['roi_volume_label'] == filter_label]
        item_label = row['el_volume']
        x_ticks.append(item_label)
        
    x_ticks = [2346,2681,3016,3351,3686,4021]
        
    plt.xticks(x_ticks, fontsize=8)
    #colors = ["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#FDAF91FF", "#AD002AFF", "#ADB6B6FF"]
    colors = ["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#AD002AFF", "#ADB6B6FF"]
    ax0.set_prop_cycle(cycler('color', colors))
    ax0.set_ylim(-0.035,0.6)
    
    markers = []
    markers.append("o")
    markers.append("^")
    markers.append("s")
    markers.append(">")
    markers.append("8")
    markers.append("D")

    ctr = 0
    for i in range(5,35, 5):
        filter_name = i
        metric_name = 'tsc_cost'
        plt.sca(ax0)
    
        label, lines = plot_mr(data, metric_name, filter_name)
        lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
        label, scatter = plot_scatter_mr(data, metric_name, filter_name, markers[ctr], colors[ctr], 25, alpha = 0.5)
        
        top_legend_handles3D_mr.append(scatter)
        top_labels3D_mr.append(label)
        ctr = ctr + 1
    
      
    # tcs by z_ score 
    ax1.set_prop_cycle(cycler('color', colors))
    ax1.set_ylabel(ur'TCS $\vert Z_{score} \vert > 2$')
    ax1.set_xlabel(r'ROI Size ' + r'$(voxel^3)$')
    ax1.tick_params(direction="in")
    ax1.set_ylim(-0.035,0.6)
    format_axes(ax1)
    
    ctr = 0
    for k in range(5,35, 5):
        filter_name = k
        metric_name = 'tsc_z_cost'
        plt.sca(ax1)
        
        label, lines = plot_mr(data, metric_name, filter_name)
        lines[0].set_linestyle(LINE_STYLES[k%NUM_STYLES])
        label, scatter = plot_scatter_mr(data, metric_name, filter_name, markers[ctr], colors[ctr], 25, alpha = 0.5)
        ctr = ctr + 1
    
    lgd = create_legend_on_bottom(3, top_legend_handles4D_mr, top_labels4D_mr, "")
    
    
    tensorDimAnn = TextArea(ur'\textbf{Tensor Dimension}' + ur' = ' + ur'$3$')
    boxUp = HPacker(children=[tensorDimAnn],
              align="center",
              pad=0, sep=5)
    
    tensorDimAnnAnchored_box = AnchoredOffsetbox(loc=8,
                                 child=boxUp, pad=0.,
                                 frameon=False,
                                 bbox_to_anchor=(-0.25, 1.03),
                                 bbox_transform=ax1.transAxes,
                                 borderpad=0.
                                 )
    
    ax1.add_artist(tensorDimAnnAnchored_box)
    
    figureA = TextArea(ur'(a) Tensor Completion Score', textprops=dict(color="k", size="x-small"))
    box = HPacker(children=[figureA],
              align="center",
              pad=0, sep=5)
    
    figureAnchored_box = AnchoredOffsetbox(loc=9,
                                 child=box, pad=0.,
                                 frameon=False,
                                 bbox_to_anchor=(0.55, -0.35),
                                 bbox_transform=ax0.transAxes,
                                 borderpad=0.
                                 )
    
    ax0.add_artist(figureAnchored_box)
    
    
    figureB = TextArea(ur'(b) Tensor Completion Score ' + ur'$\vert Z_{score}\vert > 2$', textprops=dict(color="k", size="x-small"))
    box = HPacker(children=[figureB],
              align="center",
              pad=0, sep=5)
    
    figureBAnchored_box = AnchoredOffsetbox(loc=9,
                                 child=box, pad=0.,
                                 frameon=False,
                                 bbox_to_anchor=(0.55, -0.35),
                                 bbox_transform=ax1.transAxes,
                                 borderpad=0.
                                 )
    
    ax1.add_artist(figureBAnchored_box)
    
            
    solution_id = 'structural_effect_mr_by_roi_volume3D'
    file_path = os.path.join(structural_figures, solution_id)
    texfig.savefig_pub(file_path, additional_artists=(lgd,))
    
    
def draw_3D_by_roi_volume():
    file_path = os.path.join(roi_volume_folder, data_by_mr_by_volume3D)
    data = iot.read_data_structural(file_path)

    n = 6
      
    LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
    NUM_STYLES = len(LINE_STYLES)
    
    grid_rows = 1
    grid_cols = 2
    plt.clf()
    fig, (ax0, ax1) = texfig.subplots(width=tex_width, pad = 0, nrows=grid_rows, ncols=grid_cols, sharex=True, sharey=False)
    title = ur'Effect of ROI Size on 3D fMRI Completion by Missing Values Rate  - ' + ur'Structural Missing Value Pattern';
    fig.suptitle(title, fontsize=8, fontweight='semibold')
    fig.subplots_adjust(wspace = 0.3, right = 0.95, bottom = 0.3)
    
    ax0.set_ylabel(ur'Tensor Completion Score')
    ax0.set_xlabel(ur'Missing Values (%)')
    ax0.tick_params(direction="in")
    plt.xticks(range(5, 35, 5), fontsize=8)
    
    ax0.tick_params(direction="in")
    format_axes(ax0)
    
    #colors = ["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#FDAF91FF", "#AD002AFF", "#ADB6B6FF"]
    colors = ["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#AD002AFF", "#ADB6B6FF"]
    ax0.set_prop_cycle(cycler('color', colors))
    ax0.set_ylim(-0.035,0.7)
    
    markers = []
    markers.append("o")
    markers.append("^")
    markers.append("s")
    markers.append(">")
    markers.append("8")
    markers.append("D")

    for i in range(n):
        filter_name = 'size' + str(i)
        metric_name = 'tsc_cost'
        plt.sca(ax0)
        
        label, lines = plot_volume(data, metric_name, filter_name)
        
        lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
        scatter = plot_scatter(data, metric_name, filter_name, markers[i], colors[i], 25, alpha = 0.5)
        
        top_legend_handles3D.append(lines[0])
        top_labels3D.append(label)
    
      
    # tcs by z_ score 
    ax1.set_prop_cycle(cycler('color', colors))
    ax1.set_ylabel(ur'TCS $\vert Z_{score} \vert > 2$')
    ax1.set_xlabel(ur'Missing Values (%)')
    ax1.tick_params(direction="in")
    ax1.set_ylim(-0.035,0.7)
    format_axes(ax1)
    
    for k in range(n):
        filter_name = 'size' + str(k)
        metric_name = 'tsc_z_cost'
        plt.sca(ax1)
        
        label, lines = plot_volume(data, metric_name, filter_name)
        lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
        plot_scatter(data, metric_name, filter_name, markers[k], colors[k], 25, alpha = 0.5)
    
    lgd = create_legend_on_bottom(3, top_legend_handles3D, top_labels3D, "")
    
    
    figureA = TextArea(ur'(a) Tensor Completion Score', textprops=dict(color="k", size="x-small"))
    box = HPacker(children=[figureA],
              align="center",
              pad=0, sep=5)
    
    figureAnchored_box = AnchoredOffsetbox(loc=9,
                                 child=box, pad=0.,
                                 frameon=False,
                                 bbox_to_anchor=(0.55, -0.35),
                                 bbox_transform=ax0.transAxes,
                                 borderpad=0.
                                 )
    
    ax0.add_artist(figureAnchored_box)
    
    tensorDimAnn = TextArea(ur'\textbf{Tensor Dimension}' + ur' = ' + ur'$3$')
    boxUp = HPacker(children=[tensorDimAnn],
              align="center",
              pad=0, sep=5)
    
    tensorDimAnnAnchored_box = AnchoredOffsetbox(loc=8,
                                 child=boxUp, pad=0.,
                                 frameon=False,
                                 bbox_to_anchor=(-0.25, 1.03),
                                 bbox_transform=ax1.transAxes,
                                 borderpad=0.
                                 )
    
    ax1.add_artist(tensorDimAnnAnchored_box)
    
    figureB = TextArea(ur'(b) Tensor Completion Score ' + ur'$\vert Z_{score}\vert > 2$', textprops=dict(color="k", size="x-small"))
    box = HPacker(children=[figureB],
              align="center",
              pad=0, sep=5)
    
    figureBAnchored_box = AnchoredOffsetbox(loc=9,
                                 child=box, pad=0.,
                                 frameon=False,
                                 bbox_to_anchor=(0.55, -0.35),
                                 bbox_transform=ax1.transAxes,
                                 borderpad=0.
                                 )
    
    ax1.add_artist(figureBAnchored_box)
    
    solution_id = 'structural_effect_roi_volume_by_mr_3D'
    file_path = os.path.join(structural_figures, solution_id)
    texfig.savefig_pub(file_path, additional_artists=(lgd,)) 
    

def draw_2D_by_roi_volume():
    file_path = os.path.join(roi_volume_folder, data_by_mr_by_volume2D)
    data = iot.read_data_structural(file_path)

    n = 6
      
    LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
    NUM_STYLES = len(LINE_STYLES)
    
    grid_rows = 1
    grid_cols = 2
    plt.clf()
    fig, (ax0, ax1) = texfig.subplots(width=tex_width, pad = 0, nrows=grid_rows, ncols=grid_cols, sharex=True, sharey=False)
    title = ur'Effect of ROI Size on 2D fMRI Completion by Missing Values Rate  - ' + ur'Structural Missing Value Pattern';
    fig.suptitle(title, fontsize=8, fontweight='semibold')
    fig.subplots_adjust(wspace = 0.3, right = 0.95, bottom = 0.3)
    
    ax0.set_ylabel(ur'Tensor Completion Score')
    ax0.set_xlabel(ur'Missing Values (%)')
    ax0.tick_params(direction="in")
    plt.xticks(range(5, 35, 5), fontsize=8)
    
    ax0.tick_params(direction="in")
    format_axes(ax0)
    
    #colors = ["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#FDAF91FF", "#AD002AFF", "#ADB6B6FF"]
    colors = ["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#AD002AFF", "#ADB6B6FF"]
    ax0.set_prop_cycle(cycler('color', colors))
    ax0.set_ylim(-0.035,0.7)
    
    markers = []
    markers.append("o")
    markers.append("^")
    markers.append("s")
    markers.append(">")
    markers.append("8")
    markers.append("D")

    for i in range(n):
        filter_name = 'size' + str(i)
        metric_name = 'tsc_cost'
        plt.sca(ax0)
        
        label, lines = plot_volume(data, metric_name, filter_name)
        
        lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
        scatter = plot_scatter(data, metric_name, filter_name, markers[i], colors[i], 25, alpha = 0.5)
        
        top_legend_handles2D.append(lines[0])
        top_labels2D.append(label)
    
      
    # tcs by z_ score 
    ax1.set_prop_cycle(cycler('color', colors))
    ax1.set_ylabel(ur'TCS $\vert Z_{score} \vert > 2$')
    ax1.set_xlabel(ur'Missing Values (%)')
    ax1.tick_params(direction="in")
    ax1.set_ylim(-0.035,0.7)
    format_axes(ax1)
    
    for k in range(n):
        filter_name = 'size' + str(k)
        metric_name = 'tsc_z_cost'
        plt.sca(ax1)
        
        label, lines = plot_volume(data, metric_name, filter_name)
        lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
        plot_scatter(data, metric_name, filter_name, markers[k], colors[k], 25, alpha = 0.5)
    
    lgd = create_legend_on_bottom(3, top_legend_handles2D, top_labels2D, "")
    
    
    figureA = TextArea(ur'(a) Tensor Completion Score', textprops=dict(color="k", size="x-small"))
    box = HPacker(children=[figureA],
              align="center",
              pad=0, sep=5)
    
    figureAnchored_box = AnchoredOffsetbox(loc=9,
                                 child=box, pad=0.,
                                 frameon=False,
                                 bbox_to_anchor=(0.55, -0.35),
                                 bbox_transform=ax0.transAxes,
                                 borderpad=0.
                                 )
    
    ax0.add_artist(figureAnchored_box)
    
    tensorDimAnn = TextArea(ur'\textbf{Tensor Dimension}' + ur' = ' + ur'$2$')
    boxUp = HPacker(children=[tensorDimAnn],
              align="center",
              pad=0, sep=5)
    
    tensorDimAnnAnchored_box = AnchoredOffsetbox(loc=8,
                                 child=boxUp, pad=0.,
                                 frameon=False,
                                 bbox_to_anchor=(-0.25, 1.03),
                                 bbox_transform=ax1.transAxes,
                                 borderpad=0.
                                 )
    
    ax1.add_artist(tensorDimAnnAnchored_box)
    
    figureB = TextArea(ur'(b) Tensor Completion Score ' + ur'$\vert Z_{score}\vert > 2$', textprops=dict(color="k", size="x-small"))
    box = HPacker(children=[figureB],
              align="center",
              pad=0, sep=5)
    
    figureBAnchored_box = AnchoredOffsetbox(loc=9,
                                 child=box, pad=0.,
                                 frameon=False,
                                 bbox_to_anchor=(0.55, -0.35),
                                 bbox_transform=ax1.transAxes,
                                 borderpad=0.
                                 )
    
    ax1.add_artist(figureBAnchored_box)
    
    solution_id = 'structural_effect_roi_volume_by_mr_2D'
    file_path = os.path.join(structural_figures, solution_id)
    texfig.savefig_pub(file_path, additional_artists=(lgd,))         

def draw_2D_by_roi_volume_and_mr():
    file_path = os.path.join(roi_volume_folder, data_by_volume2D)
    print ("Data Path:" + str(file_path))
    data = iot.read_data_structural(file_path)

    n = 6
     
    LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
    NUM_STYLES = len(LINE_STYLES)
    
    grid_rows = 1
    grid_cols = 2
    plt.clf()
    fig, (ax0, ax1) = texfig.subplots(width=tex_width, pad = 0, nrows=grid_rows, ncols=grid_cols, sharex=True, sharey=False)
    title = ur'Effect of Missing Values Rate on 2D fMRI Completion by ROI Size  - ' + ur'Structural Missing Value Pattern';
    fig.suptitle(title, fontsize=8, fontweight='semibold')
    fig.subplots_adjust(wspace = 0.3, right = 0.95, bottom = 0.3)
    
    ax0.set_ylabel(r'Tensor Completion Score')
    ax0.set_xlabel(r'ROI Size ' + r'$(voxel^3)$')
    
    ax0.tick_params(direction="in")
    
    format_axes(ax0)
    
    x_ticks = []
    n = 6
    for i in range(n):
        filter_label = 'size' + str(i)
        row = data.loc[data['roi_volume_label'] == filter_label]
        item_label = row['el_volume']
        x_ticks.append(item_label)
        
    x_ticks = [2346,2681,3016,3351,3686,4021]
        
    plt.xticks(x_ticks, fontsize=8)
    #colors = ["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#FDAF91FF", "#AD002AFF", "#ADB6B6FF"]
    colors = ["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#AD002AFF", "#ADB6B6FF"]
    ax0.set_prop_cycle(cycler('color', colors))
    ax0.set_ylim(-0.035,0.6)
    
    markers = []
    markers.append("o")
    markers.append("^")
    markers.append("s")
    markers.append(">")
    markers.append("8")
    markers.append("D")

    ctr = 0
    for i in range(5,35, 5):
        filter_name = i
        metric_name = 'tsc_cost'
        plt.sca(ax0)
    
        label, lines = plot_mr(data, metric_name, filter_name)
        lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
        label, scatter = plot_scatter_mr(data, metric_name, filter_name, markers[ctr], colors[ctr], 25, alpha = 0.9)
        
        top_legend_handles2D_mr.append(scatter)
        top_labels2D_mr.append(label)
        ctr = ctr + 1
    
      
    # tcs by z_ score 
    ax1.set_prop_cycle(cycler('color', colors))
    ax1.set_ylabel(ur'TCS $\vert Z_{score} \vert > 2$')
    ax1.set_xlabel(r'ROI Size ' + r'$(voxel^3)$')
    ax1.tick_params(direction="in")
    ax1.set_ylim(-0.035,0.6)
    format_axes(ax1)
    
    ctr = 0
    for k in range(5,35, 5):
        filter_name = k
        metric_name = 'tsc_z_cost'
        plt.sca(ax1)
        
        label, lines = plot_mr(data, metric_name, filter_name)
        lines[0].set_linestyle(LINE_STYLES[k%NUM_STYLES])
        label, scatter = plot_scatter_mr(data, metric_name, filter_name, markers[ctr], colors[ctr], 25, alpha = 0.5)
        ctr = ctr + 1
    
    lgd = create_legend_on_bottom(3, top_legend_handles2D_mr, top_labels2D_mr, "")
    
    
    tensorDimAnn = TextArea(ur'\textbf{Tensor Dimension}' + ur' = ' + ur'$2$')
    boxUp = HPacker(children=[tensorDimAnn],
              align="center",
              pad=0, sep=5)
    
    tensorDimAnnAnchored_box = AnchoredOffsetbox(loc=8,
                                 child=boxUp, pad=0.,
                                 frameon=False,
                                 bbox_to_anchor=(-0.25, 1.03),
                                 bbox_transform=ax1.transAxes,
                                 borderpad=0.
                                 )
    
    ax1.add_artist(tensorDimAnnAnchored_box)
    
    figureA = TextArea(ur'(a) Tensor Completion Score', textprops=dict(color="k", size="x-small"))
    box = HPacker(children=[figureA],
              align="center",
              pad=0, sep=5)
    
    figureAnchored_box = AnchoredOffsetbox(loc=9,
                                 child=box, pad=0.,
                                 frameon=False,
                                 bbox_to_anchor=(0.55, -0.35),
                                 bbox_transform=ax0.transAxes,
                                 borderpad=0.
                                 )
    
    ax0.add_artist(figureAnchored_box)
    
    
    figureB = TextArea(ur'(b) Tensor Completion Score ' + ur'$\vert Z_{score}\vert > 2$', textprops=dict(color="k", size="x-small"))
    box = HPacker(children=[figureB],
              align="center",
              pad=0, sep=5)
    
    figureBAnchored_box = AnchoredOffsetbox(loc=9,
                                 child=box, pad=0.,
                                 frameon=False,
                                 bbox_to_anchor=(0.55, -0.35),
                                 bbox_transform=ax1.transAxes,
                                 borderpad=0.
                                 )
    
    ax1.add_artist(figureBAnchored_box)
    
            
    solution_id = 'structural_effect_mr_by_roi_volume2D'
    file_path = os.path.join(structural_figures, solution_id)
    texfig.savefig_pub(file_path, additional_artists=(lgd,))

@ticker.FuncFormatter
def major_formatter(x, pos):
    return mf.format_number(x, fmt='%1.2e')

def tensor_order_by_tsc(tensor_dims, mr, roi_size, log_scale = False):
    file_path = os.path.join(roi_volume_folder,  data_all_by_volume)
    print ("Data Path:" + str(file_path))
    data = iot.read_data_structural(file_path)
    
    n = 6
     
    LINE_STYLES = ['solid', 'dashed', 'dotted']
    NUM_STYLES = len(LINE_STYLES)
    
    grid_rows = 1
    grid_cols = 1
    plt.clf()
    fig, (ax0) = texfig.subplots(width=tex_width, pad = 0, nrows=grid_rows, ncols=grid_cols, sharex=True, sharey=False)
    ur'Structural Missing Value Pattern'
    title = ur'Effect of Tensor Dimensionality on fMRI Completion by Effective ROI Size'
    fig.suptitle(title, fontsize=8, fontweight='semibold')
    fig.subplots_adjust(wspace = 0.1, left = 0.2, right = 0.95, bottom = 0.15)
    
    ax0.set_title(ur'Structural Missing Value Pattern')
    ax0.set_ylabel(ur'Tensor Completion Score')
    ax0.set_xlabel(ur'Effective ROI Size ' + ur'($voxel^3 \times timepoints$)')
    
    colors = ["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#AD002AFF", "#ADB6B6FF"]
    ax0.set_prop_cycle(cycler('color', colors))
    
    ax0.tick_params(direction="in")
    format_axes(ax0)
    
    ax0.yaxis.set_major_formatter(major_formatter)
    
    if log_scale:     
        #ax0.xscale('log')
        plt.yscale('log')
        solution_id = 'effect_tensor_dimension_roi_size_' + str(mr)+ '_log'
    else:
        solution_id = 'effect_tensor_dimension_roi_size_' + str(mr)
         
    markers = []
    markers.append("o")
    markers.append("^")
    markers.append("s")
        
    top_legend_tensor_dim = []
    top_labelstensor_dim = []
    
    ctr = 0
    for i in tensor_dims:
        plt.sca(ax0)
        filter_label = i
        subset = data.loc[data['tensor_dim1'] == filter_label]
        row = subset.loc[subset['mr'] == mr]
        
        x = row['roi_volume']
        y = row['tsc_cost']
        lw = 1
        label = ur'Tensor Dimension ' + ur'$D$' + ur' = ' + str(i)
        lines = plt.plot(x, y, linewidth=lw, label = "")
        lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
        scatter = plt.scatter(x, y, label = "",  c = colors[ctr], s = 25, marker =  markers[ctr], alpha = 1)
        
        top_legend_tensor_dim.append(scatter)
        top_labelstensor_dim.append(label)
        ctr = ctr + 1
        
    extra_text = ur'for Missing Values Rate = ' + str(mr) + ur'%'
    lgd = create_legend_on_inside_upper_left(1, top_legend_tensor_dim, top_labelstensor_dim, extra_text)
    
    file_path = os.path.join(structural_figures, solution_id)
    texfig.savefig_pub(file_path, additional_artists=(lgd,))
    
def tensor_order_by_tsc_mr(tensor_dims, roi_size, log_scale = False):
    file_path = os.path.join(roi_volume_folder,  data_all_by_volume2)
    print ("Data Path:" + str(file_path))
    data = iot.read_data_structural(file_path)
    
    n = 6
     
    LINE_STYLES = ['solid', 'dashed', 'dotted']
    NUM_STYLES = len(LINE_STYLES)
    
    grid_rows = 1
    grid_cols = 1
    plt.clf()
    fig, (ax0) = texfig.subplots(width=tex_width, pad = 0, nrows=grid_rows, ncols=grid_cols, sharex=True, sharey=False)
    ur'Structural Missing Value Pattern'
    title = ur'Effect of Tensor Dimensionality on fMRI Completion by Missing Values Rate'
    fig.suptitle(title, fontsize=8, fontweight='semibold')
    fig.subplots_adjust(wspace = 0.1, left = 0.2, right = 0.95, bottom = 0.15)
    
    ax0.set_title(ur'Structural Missing Value Pattern')
    ax0.set_ylabel(ur'Tensor Completion Score')
    ax0.set_xlabel(ur'Missing Values ' + ur'(%)')
    
    colors = ["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#AD002AFF", "#ADB6B6FF"]
    ax0.set_prop_cycle(cycler('color', colors))
    
    ax0.tick_params(direction="in")
    format_axes(ax0)
    
    ax0.yaxis.set_major_formatter(major_formatter)
    plt.xticks(range(5, 35, 5), fontsize=8)
    ax0.set_xlim(5, 35)
    
    x_ticks_map = {}
    
    x_ticks_map['size0'] = 2346
    x_ticks_map['size1'] = 2681
    x_ticks_map['size2'] = 3016
    x_ticks_map['size3'] = 3351
    x_ticks_map['size4'] = 3686
    x_ticks_map['size5'] = 4021
         
    markers = []
    markers.append("o")
    markers.append("^")
    markers.append("s")
        
    top_legend_tensor_dim = []
    top_labelstensor_dim = []
    
    ctr = 0
    
    for i in tensor_dims:
        
        
        plt.sca(ax0)
        subset = data.loc[data['tensor_dim1'] == i]
        row = subset.loc[subset['roi_volume_label'] == roi_size]
     
        x = row['mr']
        print x
        y = row['tsc_cost']
        
        roi_volume = row.tail(1)
        volume = roi_volume['el_volume']
        lw = 1
        label = ur'Tensor Dimension ' + ur'$D$' + ur' = ' + str(i)
        lines = plt.plot(x, y, linewidth=lw, label = "")
        lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
        scatter = plt.scatter(x, y, label = "",  c = colors[ctr], s = 25, marker =  markers[ctr], alpha = 1)
        
        top_legend_tensor_dim.append(scatter)
        top_labelstensor_dim.append(label)
        ctr = ctr + 1
        
        extra_text = ur'for ROI Volume = ' + str(x_ticks_map[roi_size])
    
    lgd = create_legend_on_inside_upper_left(1, top_legend_tensor_dim, top_labelstensor_dim, extra_text)
    solution_id = 'effect_tensor_dimension_by_mr_roi_size_' + str(roi_size)
    
    file_path = os.path.join(structural_figures, solution_id)
    texfig.savefig_pub(file_path, additional_artists=(lgd,))
    
def tensor_order_by_tsc_all(tensor_dims, mr, roi_size, log_scale = False):
    file_path = os.path.join(roi_volume_folder,  data_all_by_volume)
    print ("Data Path:" + str(file_path))
    data = iot.read_data_structural(file_path)
    
    n = 6
     
    LINE_STYLES = ['solid', 'dashed', 'dotted']
    NUM_STYLES = len(LINE_STYLES)
    
    grid_rows = 1
    grid_cols = 1
    plt.clf()
    fig, (ax0) = texfig.subplots(width=tex_width, pad = 0, nrows=grid_rows, ncols=grid_cols, sharex=True, sharey=False)
    ur'Structural Missing Value Pattern'
    title = ur'Effect of Tensor Dimensionality on fMRI Completion by Missing Values Rate'
    fig.suptitle(title, fontsize=8, fontweight='semibold')
    fig.subplots_adjust(wspace = 0.1, left = 0.15, right = 0.95, bottom = 0.2)
    
    ax0.set_title(ur'Structural Missing Value Pattern')
    ax0.set_ylabel(ur'Tensor Completion Score')
    ax0.set_xlabel(ur'Missing Values ' + ur'(%)')
    
    colors = ["#00468BFF", "#ED0000FF", "#42B540FF", "#0099B4FF", "#925E9FFF", "#AD002AFF", "#ADB6B6FF"]
    ax0.set_prop_cycle(cycler('color', colors))
    
    ax0.tick_params(direction="in")
    format_axes(ax0)
    
    ax0.yaxis.set_major_formatter(major_formatter)
    
    if log_scale:     
        #ax0.xscale('log')
        plt.yscale('log')
        solution_id = 'effect_tensor_dimension_roi_size_' + str(mr)+ '_log'
    else:
        solution_id = 'effect_tensor_dimension_roi_size_' + str(mr)
         
    markers = []
    markers.append("o")
    markers.append("^")
    markers.append("s")
        
    top_legend_tensor_dim = []
    top_labelstensor_dim = []
    
    ctr = 0
    for i in tensor_dims:
        plt.sca(ax0)
        filter_label = i
        subset = data.loc[data['tensor_dim1'] == filter_label]
        row = subset.loc[subset['mr'] == mr]
        
        x = row['roi_volume']
        y = row['tsc_cost']
        lw = 1.5
        label = ur'Tensor Dimension ' + ur'$D$' + ur' = ' + str(i)
        lines = plt.plot(x, y, linewidth=lw, label = "")
        lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
        scatter = plt.scatter(x, y, label = "",  c = colors[ctr], s = 25, marker =  markers[ctr], alpha = 1)
        
        top_legend_tensor_dim.append(scatter)
        top_labelstensor_dim.append(label)
        ctr = ctr + 1
        
    extra_text = ur'for Missing Values Rate = ' + str(mr) + ur'%'
    lgd = create_legend_on_inside_upper_left_inset(1, top_legend_tensor_dim, top_labelstensor_dim, extra_text,  0.01, 0.89)
    
    file_path = os.path.join(structural_figures, solution_id)
    texfig.savefig_pub(file_path, additional_artists=(lgd,))
 
def draw_structural_results():
   
    dims = [3, 4]
    for k in range(5, 35, 5):   
        #tensor_order_by_tsc(dims, k, 'size0')
        pass
    
    dims = [3, 4, 2] 
    for k in range(5, 35, 5): 
        #tensor_order_by_tsc_all(dims, k, 'size0', log_scale = True)
        pass
        
    x_ticks = ['size0', 'size1', 'size2', 'size3', 'size4']
        
    dims = [3, 4]
    
    for j in x_ticks:
        #tensor_order_by_tsc_mr(dims,  j)
        pass
    
    draw_4D_by_roi_volume()
    #draw_4D_by_roi_volume_and_mr()
    
    draw_3D_by_roi_volume()
    #draw_3D_by_roi_volume_and_mr()
    draw_2D_by_roi_volume()
    #draw_2D_by_roi_volume_and_mr()


if __name__ == "__main__":
    draw_structural_results()