from bokeh.plotting import figure, show, save
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.io import export_svg
from bokeh.transform import cumsum
from math import pi
import json
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import os

SDG_COLORS = {
    'ODS1':(229, 35, 61),
    'ODS2':(221, 167, 58),
    'ODS3':(76, 161, 70),
    'ODS4':(197, 25, 45),
    'ODS5':(239, 64, 44),
    'ODS6':(39, 191, 230),
    'ODS7':(251, 196, 18),
    'ODS8':(163, 28, 68),
    'ODS9':(242, 106, 45),
    'ODS10':(224, 20, 131),
    'ODS11':(248, 157, 42),
    'ODS12':(191, 141, 44),
    'ODS13':(64, 127, 70),
    'ODS14':(31, 151, 212),
    'ODS15':(89, 186, 72),
    'ODS16':(18, 106, 159),
    'ODS17':(19, 73, 107)
}


with open('./data/resultados/final_abstracts_classifications2019.json' , 'r') as fp:
    DATA_2019 = json.load(fp)
with open('./data/resultados/final_abstracts_classifications2020.json' , 'r') as fp:
    DATA_2020 = json.load(fp)
with open('./data/resultados/final_abstracts_classifications2021.json' , 'r') as fp:
    DATA_2021 = json.load(fp)
with open('./data/resultados/final_abstracts_classifications2022.json' , 'r') as fp:
    DATA_2022 = json.load(fp)
with open('./data/resultados/final_abstracts_classifications2023.json' , 'r') as fp:
    DATA_2023 = json.load(fp)
with open('./data/resultados/final_abstracts_classificationsTOTAL.json' , 'r') as fp:
    DATA_TOTAL = json.load(fp)

def total_metrics_bars():
    data = {}
    x = []
    metrics = []
    colors = []
    for ods in DATA_TOTAL:
        x.append(ods)
        ods_data = DATA_TOTAL[ods]
        metrics.append(ods_data)
        colors.append(SDG_COLORS[ods])
    data = dict(x=x, counts=metrics, color=colors)
    source = ColumnDataSource(data=data)
    p = figure(x_range=FactorRange(*x), 
           height=350, 
           width= 700,
        #    title="Número de publicaciones por objetivo",
           toolbar_location=None, 
           tools="")
    
    p.vbar(x='x', top='counts', width=0.5, source=source, color='color')
    p.xaxis.axis_label = 'ODS'
    p.yaxis.axis_label = 'Número de publicaciones'
    # p.group_padding = 0.1
    # p.title.text_font_style = "bold"
    # p.title.text_font_size = "25px"
    # p.title.text_color = (0, 58, 112)

    filename = f'metricas_totales_generales'

    service = Service()
    options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(service=service, options=options)
    p.output_backend = "svg"
    export_svg(p, filename=f'./temp/{filename}.svg', webdriver=driver, height=350, width= 700)

    os.popen(f'inkscape ./temp/{filename}.svg --export-filename ./memoria/imagenes/{filename}.eps')
    # return p

def sdg_progress(sdg:int=1):
    for ods_num in range(1,18):
        sdg = f'ODS{ods_num}'
        years = ['2019', '2020', '2021', '2022', '2023']
        data_2019 = DATA_2019[sdg]
        data_2020 = DATA_2020[sdg]
        data_2021 = DATA_2021[sdg]
        data_2022 = DATA_2022[sdg]
        data_2023 = DATA_2023[sdg]
        y_data = [data_2019, data_2020, data_2021, data_2022, data_2023]
        p = figure(height=350, 
                width=650, 
                x_range=['2019', '2020', '2021', '2022', '2023'], 
                tools="", 
                toolbar_location=None)
        p.line(years, y_data, line_width=2, color=SDG_COLORS[sdg])
        p.circle(years, y_data, fill_color='white',color=SDG_COLORS[sdg], size=9, line_width=2.7)
        p.xaxis.axis_label = 'Año'
        p.yaxis.axis_label = 'Número de publicaciones'

        # p.title.text_font_style = "bold"
        # p.title.text_font_size = "25px"
        # p.title.text_color = (0, 58, 112)

        filename = f'EvolucionOds{ods_num}'

        service = Service()
        options = webdriver.ChromeOptions()
        driver = webdriver.Chrome(service=service, options=options)

        p.output_backend = "svg"
        export_svg(p, filename=f'./temp/{filename}.svg', webdriver=driver, height=350, width=650)

        os.popen(f'inkscape ./temp/{filename}.svg --export-filename ./memoria/imagenes/{filename}.eps')

def queso_ods():
    total_classified = 0
    for sdg in DATA_TOTAL:
        total_classified += DATA_TOTAL[sdg]
    angles = []
    for sdg in DATA_TOTAL:
        angles.append((DATA_TOTAL[sdg]/total_classified * (2*pi*-1)))

    colors = []
    sdgs = []
    for sdg in SDG_COLORS:
        colors.append(SDG_COLORS[sdg])
        sdgs.append(sdg)

    data = dict(angle=angles, color=colors, sdg=sdgs)

    p = figure(height=500, 
                 width=650, 
                #  title="Porcentaje de articulos por objetivo",
                 tools="", 
                 toolbar_location=None, 
                 x_range=(-0.5, 1.0))#,
                #  y_range=(1,-1))
    
    p.annular_wedge(x=0.1, y=1,  
                      inner_radius=0.3,  
                      outer_radius=0.5, 
                      direction='clock',
                      start_angle=cumsum('angle', include_zero=True), 
                      end_angle=cumsum('angle'),
                      line_color="white", 
                      line_width=5,
                      fill_color='color', 
                      legend_field='sdg', 
                      source=data)
    
    p.axis.axis_label=None
    p.axis.visible=False
    p.grid.grid_line_color = None
    p.outline_line_color = None
    p.legend.label_text_font_size = "20px"
    # p.title.text_font_style = "bold"
    # p.title.text_font_size = "25px"
    # p.title.text_color = (0, 58, 112)

    filename = f'resultados_queso'

    service = Service()
    options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(service=service, options=options)

    p.output_backend = "svg"
    export_svg(p, filename='./temp/resultados_queso.svg', webdriver=driver)#,height=500, width=650)
    os.popen(f'inkscape ./temp/{filename}.svg --export-filename ./memoria/imagenes/{filename}.eps')

def main():
    pass

if __name__ == "__main__":
    # queso_ods()
    sdg_progress()
    # total_metrics_bars()