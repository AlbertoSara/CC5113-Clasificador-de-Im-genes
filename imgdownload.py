# -*- coding: utf-8 -*-
from urllib.request import urlretrieve
import csv


with open('datos_imagenes_propiedades.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    counter = 0
    failed = []
    for row in csv_reader:
        try:
            urlretrieve(row[0],"imagenes proyecto/" + str(counter) + ".jpg");
            counter += 1
            print(str(100.0*(counter)/43237.0) +" "+ str(counter))
        except KeyboardInterrupt:
            failed.append((row[0],counter))
            break
        except:
            failed.append((row[0],counter))
            print("Error: " + str(counter))
    while len(failed) > 0:
        for (row,rownum) in failed:
            try:
                failed.remove((row,rownum))
                urlretrieve(row,"imagenes proyecto/"+ str(rownum) + ".jpg");
                counter += 1
                print("Previously failed: " + str(100.0*(counter)/(43237.0)) + " " + str(rownum))
            except KeyboardInterrupt:
                break
            except:
                failed.append((row,rownum))
                print("Error: " + str(rownum))
