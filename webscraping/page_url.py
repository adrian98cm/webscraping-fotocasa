# Import libraries
import requests
from bs4 import BeautifulSoup
import csv
import time
from home import Home

def scrap_page(page_url, district):
    # Global variables
    url = page_url
    time.sleep(5)

    # Getting whole information (HTML) in page_url
    page = requests.get(url).text
    soup = BeautifulSoup(page, 'html.parser')

    with open('buildings_information.csv', 'a', newline='') as csv_file:
        headers = ['Precio', 'Distrito','Tipo de inmueble', 'Habitaciones', 'Aseos', 'Superficie', 'Planta', 'Parking', 'URL']
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        #writer.writeheader()
        
        home = Home(url, district)
        
        # Searching information
        # Price
        price = soup.find('span', attrs = { 'class':'re-DetailHeader-price' }).string
        b = slice(len(price)-2)
        home.price = price[b]

        # Header
        header = soup.findAll('li', attrs = { 'class': 're-DetailHeader-featuresItem'})
        for i in header:
            element = i.findAll('span')
            tmp = element[len(element)-2].get_text().split(' ')
            number = tmp[0]
            if len(tmp) != 1:
                type = tmp[1]
                if type == 'habs.':
                    home.rooms = number
                if type == 'hab.':
                    home.rooms = number
                if type == 'baños':
                    home.baths = number
                if type == 'baño':
                    home.baths = number
                if type.startswith('m'):
                    home.size = number
                if type == 'Planta':
                    a = slice(len(number)-1)
                    home.floor = number[a]
            else:
                number = element[len(element)-1].get_text()

        # Characteristics
        characteristics = soup.findAll('div', attrs = { 'class': 're-DetailFeaturesList-feature'})   
        type = characteristics[0].find('p', attrs = { 'class': 're-DetailFeaturesList-featureLabel'}).get_text()
        valueType = characteristics[0].find('p', attrs = { 'class': 're-DetailFeaturesList-featureValue'}).get_text()
        for i in characteristics:
            type = i.find('p', attrs = { 'class': 're-DetailFeaturesList-featureLabel'}).get_text()
            valueType = i.find('p', attrs = { 'class': 're-DetailFeaturesList-featureValue'}).get_text()
        
            if type == 'Tipo de inmueble':
                home.type = valueType
            if type == 'Parking':
                home.parking = 1

        home.toString()
        writer.writerow({'Precio':           home.price, 
                         'Distrito':         home.district,
                         'Tipo de inmueble': home.type,
                         'Habitaciones':     home.rooms,
                         'Aseos':            home.baths, 
                         'Superficie':       home.size, 
                         'Planta':           home.floor,
                         'Parking':          home.parking,
                         'URL':              home.url})