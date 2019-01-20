from selenium import webdriver
import time
from Car import Car
from selenium.common.exceptions import NoSuchElementException
import pandas as pd
import csv


base_url = "https://www.finn.no/car/used/search.html"
details_url = "https://www.finn.no/car/used/ad.html?finnkode="
xpath_common = "/html/body/main/div[3]/div[1]/div/section["
xpath_text = "]/dl/dt["
xpath_value = "]/dl/dd["
pagination_url = base_url + "?page="
pagination_start_value = 242
pagination_end_value = 1000
ads_per_page = 50

chrome_options = webdriver.ChromeOptions()

browser = webdriver.Chrome(chrome_options=chrome_options)
browser.set_window_size(360, 640)
browser.get(base_url)

for y in range(pagination_start_value, pagination_end_value):
    print(y)
    cars = []
    finn_codes_list = []
    finn_codes = browser.find_elements_by_xpath("//*[starts-with(@id, 'save-favorite-')]")
    assert len(finn_codes) > 0, "no results"
    for e in finn_codes:
        finn_codes_list.append(e.get_attribute('data-heart-ad-id')) #Extract "finn-kode" from each advert displayed on current page.

    finn_codes_list.pop(0) #Remove paid ad

    #Har tallene stokket seg litt?
    #Prøv å skrive FINN-koden (8 eller 9 sifre) på nytt

    for e in finn_codes_list:
        browser.get(details_url + e)

        try:
            price = browser.find_element_by_xpath("/html/body/main/div[3]/div[1]/div/section[1]/div[1]/span[2]").text
        except NoSuchElementException:
            continue
        try:
            model = browser.find_element_by_xpath("/html/body/main/div[3]/div[1]/div/div[6]/h1").text
        except NoSuchElementException:
            model = browser.find_element_by_xpath("/html/body/main/div[3]/div[1]/div/div[4]/h1").text

        model_year = ""
        first_reg = ""
        km = ""
        color = ""
        gear = ""
        hjuldrift = ""
        drivstoff = ""
        effekt = ""
        sylindervolum = ""
        for x in range(2, 5):
            for i in range(1, 15):
                try:
                    if "Årsmodell" in browser.find_element_by_xpath(xpath_common+str(x)+xpath_text+str(i)+"]").text:
                        model_year = browser.find_element_by_xpath(xpath_common+str(x)+xpath_value+str(i)+"]").text
                        continue
                    if "gang registrert" in browser.find_element_by_xpath(xpath_common+str(x)+xpath_text+str(i)+"]").text:
                        first_reg = browser.find_element_by_xpath(xpath_common+str(x)+xpath_value+str(i)+"]").text
                        continue
                    if "Km.stand" in browser.find_element_by_xpath(xpath_common+str(x)+xpath_text+str(i)+"]").text:
                        km = browser.find_element_by_xpath(xpath_common+str(x)+xpath_value+str(i)+"]").text
                        continue
                    if "Farge" in browser.find_element_by_xpath(xpath_common+str(x)+xpath_text+str(i)+"]").text:
                        color = browser.find_element_by_xpath(xpath_common+str(x)+xpath_value+str(i)+"]").text
                        continue
                    if "Girkasse" in browser.find_element_by_xpath(xpath_common+str(x)+xpath_text+str(i)+"]").text:
                        gear = browser.find_element_by_xpath(xpath_common+str(x)+xpath_value+str(i)+"]").text
                        continue
                    if "Hjuldrift" in browser.find_element_by_xpath(xpath_common+str(x)+xpath_text+str(i)+"]").text:
                        hjuldrift = browser.find_element_by_xpath(xpath_common+str(x)+xpath_value+str(i)+"]").text
                        continue
                    if "Drivstoff" in browser.find_element_by_xpath(xpath_common+str(x)+xpath_text+str(i)+"]").text:
                        drivstoff = browser.find_element_by_xpath(xpath_common+str(x)+xpath_value+str(i)+"]").text
                        continue
                    if "Effekt" in browser.find_element_by_xpath(xpath_common+str(x)+xpath_text+str(i)+"]").text:
                        effekt = browser.find_element_by_xpath(xpath_common+str(x)+xpath_value+str(i)+"]").text
                        continue
                    if "Sylindervolum" in browser.find_element_by_xpath(xpath_common+str(x)+xpath_text+str(i)+"]").text:
                        sylindervolum = browser.find_element_by_xpath(xpath_common+str(x)+xpath_value+str(i)+"]").text
                        continue
                except NoSuchElementException:
                    pass

        car = Car(price, model, model_year, first_reg, km, color, gear, hjuldrift, drivstoff, effekt, sylindervolum, e)

        df = pd.DataFrame.from_records([car.to_dict()])
        with open('../cars.csv', 'a') as f:
            df.to_csv(f, header=f.tell() == 0, index=0)



    browser.get(pagination_url+str(y)) #Next page


browser.quit()
