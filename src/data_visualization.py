from bokeh.plotting import figure, show, save
from bokeh.models import Legend, Label
from bokeh.io import export_svg
import json
import cairosvg
import shutil

# Color pallete: https://coolors.co/003a70-ffc72c-212121-c13b32-4f9951-2572d0-d39417

for model in range(1,27):
    # if model != 2: continue
    with open(f'./models/model{model}/metrics.json') as fp:    
        metrics = json.load(fp)
    thresholds = []
    accuracy = []
    f1_s = []
    precission = []
    recall = []
    for metric in metrics:
        thresholds.append(metric['Threshold'])
        accuracy.append(metric['Accuracy'])
        precission.append(metric['Precission'])
        recall.append(metric['Recall'])
        f1_s.append(metric['F1Score'])
    
    # title=f"Model{model} metrics"
    plot = figure(x_axis_label='Umbral', y_axis_label='Valor', height=500, width=650, x_range=(0.1, 0.9), tools="", toolbar_location=None)

    plot.ygrid.grid_line_color = None

    ticks = [0, 0.1, 0.3, 0.5, 0.7, 0.9]
    plot.xaxis[0].ticker = ticks
    plot.xgrid[0].ticker = ticks

    plot.xgrid.band_hatch_pattern   = "/"
    plot.xgrid.band_hatch_alpha     = 0.5
    plot.xgrid.band_hatch_color     = (70,70,70)
    plot.xgrid.band_hatch_weight    = 0.5
    plot.xgrid.band_hatch_scale     = 15

    acc_plot    = plot.line(thresholds, accuracy,   line_width=2,   color=(193, 59, 50),   )#legend_label="Accuracy.",  )
    prec_plot   = plot.line(thresholds, precission, line_width=2,   color=(211, 148, 23),    )#legend_label="Precission.",)
    recall_plot = plot.line(thresholds, recall,     line_width=2,   color=(37, 114, 208),  )#legend_label="Recall.",    )
    f1_plot     = plot.line(thresholds, f1_s,       line_width=2,   color=(79, 153, 81), )#legend_label="F1Score.",   )

    legend = Legend(items=[
    ("Exactitud"     , [acc_plot]),
    ("Precision"     , [prec_plot]),
    ("Exhaustibidad" , [recall_plot]),
    ("Valor-F1"      , [f1_plot]),
    ], location="right"
    ,  orientation = "horizontal")

    plot.add_layout(legend, 'above')

    # plot.title.text_font_style = "bold"
    # plot.title.text_font_size = "25px"
    # plot.title.text_color = (0, 58, 112)

    label_x = 0.2
    try:
        accuracy_label_y    = accuracy[2]      + 0.01
        precission_label_y  = precission[2]    + 0.01
        recall_label_y      = recall[2]        + 0.01
        f1_s_label_y        = f1_s[2]          + 0.01
    except: 
        print(model)
        accuracy_label = 0.5
        
    accuracy_label              = Label(x = label_x, y = accuracy_label_y,   text = 'Exactitud',     text_color=(33, 33, 33), x_units='data', y_units='data', text_font_size = '10px', text_font_style = 'bold')
    precission_label_y_label    = Label(x = label_x, y = precission_label_y, text = 'Precision',     text_color=(33, 33, 33), x_units='data', y_units='data', text_font_size = '10px', text_font_style = 'bold')
    recall_label_y_label        = Label(x = label_x, y = recall_label_y,     text = 'Exhaustibidad', text_color=(33, 33, 33), x_units='data', y_units='data', text_font_size = '10px', text_font_style = 'bold')
    f1_s_label_y_label          = Label(x = label_x, y = f1_s_label_y,       text = 'Valor-F1',      text_color=(33, 33, 33), x_units='data', y_units='data', text_font_size = '10px', text_font_style = 'bold')

    plot.add_layout(accuracy_label)
    plot.add_layout(precission_label_y_label)
    plot.add_layout(recall_label_y_label)
    plot.add_layout(f1_s_label_y_label)
    # plot.title.offset = -100
    # plot.legend.orientation = "horizontal"

    plot.output_backend = "svg"
    export_svg(plot, filename=f'./models/model{model}/metrics.svg')
    cairosvg.svg2eps(url=f'./models/model{model}/metrics.svg', write_to=f'./models/model{model}/metrics.eps')
    pdf_image_path = f'./memoria/imagenes/model{model}_metrics.eps'
    shutil.copy(f'./models/model{model}/metrics.eps', pdf_image_path)
    # exit()
