########################################################
#        Program to Download Wallpapers from           #
#                  alpha.wallhaven.cc                  #
#                                                      #
#                 Author - Saurabh Bhan                #
#                                                      #
#                  Dated- 26 June 2016                 #
#                 Update - 11 June 2019                #
########################################################

import os
import getpass
import re
import requests
import tqdm
import time
import urllib
import json
from bs4 import BeautifulSoup
from collections import OrderedDict
from pprint import pprint

os.makedirs('Wallhaven', exist_ok=True)
BASEURL=""
cookies=dict()

global APIKEY
APIKEY = ""

def category():
    global BASEURL
    print('''
    ****************************************************************
                            Category Codes

    all     - Every wallpaper.
    general - For 'general' wallpapers only.
    anime   - For 'Anime' Wallpapers only.
    people  - For 'people' wallapapers only.
    ga      - For 'General' and 'Anime' wallapapers only.
    gp      - For 'General' and 'People' wallpapers only.
    ****************************************************************
    ''')
    ccode = input('Enter Category: ').lower()
    ctags = {'all':'111', 'anime':'010', 'general':'100', 'people':'001', 'ga':'110', 'gp':'101' }
    ctag = ctags[ccode]

    print('''
    ****************************************************************
                            Purity Codes

    sfw     - For 'Safe For Work'
    sketchy - For 'Sketchy'
    nsfw    - For 'Not Safe For Work'
    ws      - For 'SFW' and 'Sketchy'
    wn      - For 'SFW' and 'NSFW'
    sn      - For 'Sketchy' and 'NSFW'
    all     - For 'SFW', 'Sketchy' and 'NSFW'
    ****************************************************************
    ''')
    pcode = input('Enter Purity: ')
    ptags = {'sfw':'100', 'sketchy':'010', 'nsfw':'001', 'ws':'110', 'wn':'101', 'sn':'011', 'all':'111'}
    ptag = ptags[pcode]

    BASEURL = 'https://wallhaven.cc/api/v1/search?apikey=' + APIKEY + "&categories=" +\
        ctag + '&purity=' + ptag + '&page='
    return 'random'

def latest():
    global BASEURL
    print('Downloading latest')
    topListRange = '1M'
    BASEURL = 'https://wallhaven.cc/api/v1/search?apikey=' + APIKEY + '&topRange=' +\
    topListRange + '&sorting=toplist&page='
    return 'latest'

def search():
    global BASEURL
    query = input('Enter search query: ')
    BASEURL = 'https://wallhaven.cc/api/v1/search?apikey=' + APIKEY + '&q=' + \
        urllib.parse.quote_plus(query) + "&categories=" +\
        '100' + '&purity=' + '100' +'&page='
    return query
def stuborn_request(url, sleep_time=0.5):
    try: 
        urlreq = requests.get(url, cookies=cookies)
        while(urlreq.status_code==429):
            #print(f"Too many requests... Pausing for {sleep_time} seconds.")
            time.sleep(sleep_time)
            urlreq = requests.get(url, cookies=cookies)
    except Exception as e:
        print(e)
        time.sleep(5)
        urlreq = stuborn_request(url, sleep_time*2)

    time.sleep(0.1)
    return urlreq

def getTags(img_url, as_list=False):
    urlreq = stuborn_request(img_url+'?apikey=' + APIKEY)
    
    soup = BeautifulSoup(urlreq.content, 'html.parser')
    title = soup.title.string
    tag_string = title.split("|")[0] 
    if as_list:
        return tag_string.split(',')
    else:
        return tag_string


def downloadPage(pageId, totalImage, path="Wallhaven", small=False):
    url = BASEURL + str(pageId)
    urlreq = stuborn_request(url)
    
    pagesImages = json.loads(urlreq.content);
    pageData = pagesImages["data"]

    with open(os.path.join(path, "labels.txt"), 'a')as label_file:

        for i in range(len(pageData)):
            currentImage = (((pageId - 1) * 24) + (i + 1))
            name = pageData[i]['id']
            if small is False:
                url = pageData[i]["path"]
            else:
                url = pageData[i]['thumbs']['small']
            

            filename = os.path.basename(url)
            osPath = os.path.join(path, filename)
            tags = getTags(pageData[i]["url"])
            label_file.write(name + ' | '+ tags + '\n')
            if not os.path.exists(osPath):
                imgreq = stuborn_request(url)

                if imgreq.status_code == 200:
                    print("Downloading : %s - %s / %s" % (filename, currentImage , totalImage))
                    with open(osPath, 'ab') as imageFile:
                        for chunk in imgreq.iter_content(1024):
                            imageFile.write(chunk)
                elif (imgreq.status_code != 403 and imgreq.status_code != 404):
                    print("Unable to download %s - %s / %s" % (filename, currentImage , totalImage))
            else:
                print("%s already exist - %s / %s" % (filename, currentImage , totalImage))

def main():
    # Choice = input('''Choose how you want to download the image:

    #Enter "category" for downloading wallpapers from specified categories
    #Enter "latest" for downloading latest wallpapers
    #Enter "search" for downloading wallpapers from search

    #Enter choice: ''').lower()'''
    Choice ='search'
    while Choice not in ['category', 'latest', 'search']:
        if Choice != None:
            print('You entered an incorrect value.')
        choice = input('Enter choice: ')

    if Choice == 'category':
        searching = category()
    elif Choice == 'latest':
        searching = latest()
    elif Choice == 'search':
        searching = search()

    pgid = int(input('How Many pages you want to Download: '))
    totalImageToDownload = str(24 * pgid)
    new_path =  os.path.join("Wallhaven", searching)
    os.makedirs(new_path, exist_ok=True)
    print('Number of Wallpapers to Download: ' + totalImageToDownload)
    for j in range(1, pgid + 1):
        downloadPage(j, totalImageToDownload, new_path, small=True)

if __name__ == '__main__':
    main()
