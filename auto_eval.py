import os
import sys
import argparse
from standardization import standard
from threaded_final import mainplot
from evaluate_final1 import evaluate

color_markers= ['Magenta','Neon Green','Green','Blue']

ap = argparse.ArgumentParser()
ap.add_argument("-ref", "--reference", required=True,
                help="Path to mp4 or mov video")
ap.add_argument("-fol", "--folder", required=True,
	help="path to the folder where videos are stored")
ap.add_argument("-cm_top", "--colormarker_top", required=False, default=0,type=int,
	help="0 - Magenta, 1 - Neon Green , 2 - Green , 3 - Blue"
         "Default Value  = 0")
ap.add_argument("-cm_phys", "--colormarker_physical", required=False, default=3,type=int,
    help="-1 - None ,0 - Magenta, 1 - Neon Green , 2 - Green , 3 - Blue"
         "Default Value  = 3")

ap.add_argument("-wid", "--width", required=False, default=20, type=int,
	help="Any integer value between 10 to 40")
ap.add_argument("-weight", "--weightage", required=False, nargs="*",type=int, default=[10,70,0,10,10],
	help="A tuple of weight given to each technique of evaluation")

#ap.add_argument("-cm_phys", "--colormarker_physical", required=False,help="0 - Magenta, 1 - Neon Green , 2 - Green , 3 - Blue")
args = vars(ap.parse_args())

# display a friendly message to the user

perfect_file_name = args["reference"]
videos_folder_path = args["folder"]
color_marker_top = args["colormarker_top"]
color_marker_physical =args["colormarker_physical"]
width = args["width"]
weightage = args["weightage"]

print("You have chosen the standard file :"+str(perfect_file_name))
print("You have chosen the folder's path :"+str(videos_folder_path))
print("You have chosen the top color marker as :"+str(color_markers[color_marker_top]))
print("You have chosen the physical color marker as :"+str(color_markers[color_marker_physical]))
print("You have chosen the width as :"+str(width))
if weightage.__len__()!=5 and sum(weightage)!=100:
    print("Sum of the 5 methods should be 100")
else:
    print("You have chosen the weightage as :"+str(weightage))

    print("Do you want to start execution? Y/N")
    choice = input()
    if choice =='Y':
        print("welcome")
        csv_file_name = standard(perfect_file_name, width)

        mainplot(csv_file_name, videos_folder_path, color_marker_top, color_marker_physical)

        #evaluate(csv_file_name)

    else:
        exit(-1)
