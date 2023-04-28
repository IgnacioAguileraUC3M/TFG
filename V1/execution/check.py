import pandas as pd
import keyboard
import time
data = pd.read_csv('./scopus(1).csv')
final_data = 'Text\n'
counter = 0

for text in data['Abstract']:
    print(text)
    pressed = False
    key = None
    while not pressed:
        if keyboard.is_pressed('y'):
            pressed = True
            key = 'y'
            counter += 1
        elif keyboard.is_pressed('n'):
            pressed = True
            key = 'n'
    match key:
        case 'y':
            final_data += '"'+text+'"\n'
        case 'n':
            pass
    if counter > 5:
        break
    print('=====================================')
    print('=====================================')
    time.sleep(1)

with open('./test_data.csv', 'w') as fp:
    fp.write(final_data)

            
