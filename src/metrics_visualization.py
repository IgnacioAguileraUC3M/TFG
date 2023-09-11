from bokeh.plotting import figure, show, save
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.models import Legend, Label
from bokeh.io import export_svg
import json
import cairosvg
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import os


metrics_path = './data/resultados/f1_metrics.json'
accuracies = []
with open(metrics_path, 'r') as fp:
    metrics = json.load(fp) 

models = []
colors = []
activations = []
for model in metrics:
    if int(model) < 11: continue
    if int(model) <= 18:
        data_used = 'Datos iniciales' 
    elif int(model) < 22:
        data_used = 'Datos aumentados'
    elif int(model) == 22:
        data_used = 'Datos de iisd'
    else:
        data_used = 'Datos finales'

    bar_color = (0, 58, 112) if int(model) <= 15 else (224, 175, 38)
    activation = 'softmax' if int(model) <= 15 else 'sigmoid'
    colors.append(bar_color)
    activations.append(activation)

    accuracies.append(metrics[model])
    models.append((data_used, model))

source = ColumnDataSource(data=dict(x=models, counts=accuracies, color=colors, activation=activations))

p = figure(x_range=FactorRange(*models), 
           height=350, 
        #    title="Exactitud por modelo",
           toolbar_location=None, 
           tools="")#,
        #    x_axis_label='Model', 
        #    y_axis_label='Accuracy')

p.vbar(x='x', top='counts', color='color', legend_field='activation', width=0.9, source=source)

p.y_range.start = 0.0
p.x_range.range_padding = 0.1
# p.x_range.group_padding = 5
p.xaxis.major_label_orientation = 1
p.xaxis.group_label_orientation = 0.2
p.xaxis.group_text_align = 'right'
p.xgrid.grid_line_color = None
p.legend.orientation = "horizontal"
p.legend.location = "top_left"

# p.title.text_font_style = "bold"
# p.title.text_font_size = "25px"
# p.title.text_color = (0, 58, 112)

p.output_backend = "svg"

service = Service()
options = webdriver.ChromeOptions()
driver = webdriver.Chrome(service=service, options=options)
export_svg(p, filename=f'./temp/models_f1_metrics.svg', webdriver=driver)
os.popen(f'inkscape ./temp/models_f1_metrics.svg --export-filename ./memoria/imagenes/models_f1_metrics.eps')
# show(p)
# cairosvg.svg2eps(url=f'./v1/model/models_metrics.svg', write_to=f'./v1/model/models_metrics.eps')