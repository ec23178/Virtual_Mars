import math
import numpy as np
import os
import xml.etree.ElementTree as ET

def convert(dataFolder):
    
    counter = 1
    width = ""
    height = ""
    namespaces = {'geom' : 'http://pds.nasa.gov/pds4/geom/v1'}
    os.makedirs("COLMAP_params", exist_ok=True)

    for imageFile in os.listdir(dataFolder):

        # Read all contents of the .xml file.
        tree = ET.parse(dataFolder + '/' + imageFile)
        root = tree.getroot()

        # FOR .xml FILES:

        for vector in root.findall(".//{http://pds.nasa.gov/pds4/geom/v1}Vector_Axis", namespaces):
            a_x = vector.find("./geom:x_unit", namespaces) # x
            a_y = vector.find("geom:y_unit", namespaces) # y
            a_z = vector.find("geom:z_unit", namespaces) # z

            a = [float(a_x.text), float(a_y.text), float(a_z.text)]

            # print(a)


        for vector in root.findall(".//{http://pds.nasa.gov/pds4/geom/v1}Vector_Horizontal", namespaces):
            h_x = vector.find("./geom:x_pixel", namespaces) # x
            h_y = vector.find("geom:y_pixel", namespaces) # y
            h_z = vector.find("geom:z_pixel", namespaces) # z

            h = [float(h_x.text), float(h_y.text), float(h_z.text)]

            # print(h)


        for vector in root.findall(".//{http://pds.nasa.gov/pds4/geom/v1}Vector_Vertical", namespaces):
            v_x = vector.find("./geom:x_pixel", namespaces) # x
            v_y = vector.find("geom:y_pixel", namespaces) # y
            v_z = vector.find("geom:z_pixel", namespaces) # z

            v = [float(v_x.text), float(v_y.text), float(v_z.text)]

            # print(v)


        # Print vectors a, h and v. (Testing)
        # print(a, h, v)

        # Use the Mathematical formulas presented in 
        # https://pmc.ncbi.nlm.nih.gov/articles/PMC7892537/pdf/11214_2021_Article_795.pdf , page 58
        # to calculate fx, fy, px, py

        fx = np.dot(a,h)
        fy = np.dot(a,v)
        px0 = np.cross(a,h)
        px = math.sqrt((px0[0] ** 2) + (px0[1] ** 2) + (px0[2] ** 2))
        py0 = np.cross(a,v)
        py = math.sqrt((py0[0] ** 2) + (py0[1] ** 2) + (py0[2] ** 2))

        # Take absolute values of fx and fy, and calculate average f for the SIMPLE_RADIAL model.
        f = (abs(fx) + abs(fy))/2

        # Print all calculated parameters. (Testing)
        # print(fx, fy, px, py, f)
        # print(px0)
        

        # Convention for feature input in COLMAP is CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[] separated by spaces
        # Here PARAMS[] will be: f, px, py, where f is the average of fx and fy.
        # SIMPLE_RADIAL MODEL chosen as per documentation: https://colmap.github.io/cameras.html
        # Write calculated values into new text file following COLMAP conventions.

        # Get name of .xml file.
        path = 'data/' + imageFile
        file_name_full = os.path.basename(path)
        file_name = os.path.splitext(file_name_full)[0]

        # Width and height of all images in pixels
        if file_name[1] == "L":
            width = "1270"
            height = "488"
        else:
            width = "793"
            height = "709"

        # Name new text file with parameters using same name in xml file.
        # Write to corresponding .txt file in COLMAP_params
        newFile = open("COLMAP_params/" + str(file_name) + ".txt", "a")
        newFile.write(str(counter) + " SIMPLE_RADIAL " + width + " " + height + " " + str(f) + ", " + str(px) + ", " + str(py) + ", 0")

        # Append to param_list .txt file to have a list for all images.
        allParam = open("cameras.txt", "a")
        allParam.write(str(counter) + " SIMPLE_RADIAL " + width + " " + height + " " + str(f) + ", " + str(px) + ", " + str(py) + ", 0\n" + str(file_name) + "\n")

        counter += 1
        newFile.close
        allParam.close


    print("CAHVOR to COLMAP conversion complete!")
    return

if __name__ == '__main__':
    convert("data")