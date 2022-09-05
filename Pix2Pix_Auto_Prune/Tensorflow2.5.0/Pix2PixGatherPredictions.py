import xlsxwriter
import os
workbook = xlsxwriter.Workbook("C:\\Users\\Woody\\Desktop\\Test\\All_Results.xlsx")

base_dir = "C:\\Users\\Woody\\Desktop\\Test\\"
file = "evaluation_results.txt"

for sub_dir in os.listdir(base_dir):
    for fin_dir in os.listdir(base_dir+sub_dir):
        print(base_dir+sub_dir+"\\"+fin_dir+"\\"+file)

        file_loc = base_dir+sub_dir+"\\"+fin_dir+"\\"+file


        result_file = open(file_loc, 'r')
        lines = result_file.readlines()

        look_for = ["Mean pixel accuracy: ",
                    "Mean class accuracy: ",
                    "Mean class IoU: ",
                    "road           : acc = ", 
                    "sidewalk       : acc = ", 
                    "building       : acc = ", 
                    "wall           : acc = ", 
                    "fence          : acc = ", 
                    "pole           : acc = ", 
                    "traffic light  : acc = ", 
                    "traffic sign   : acc = ", 
                    "vegetation     : acc = ", 
                    "terrain        : acc = ", 
                    "sky            : acc = ", 
                    "person         : acc = ", 
                    "rider          : acc = ", 
                    "car            : acc = ", 
                    "truck          : acc = ", 
                    "bus            : acc = ", 
                    "train          : acc = ", 
                    "motorcycle     : acc = ", 
                    "bicycle        : acc = "]

        worksheet = workbook.add_worksheet(sub_dir+fin_dir)

        worksheet.write('A1', 'Mean pixel accuracy')
        worksheet.write('A2', 'Mean class accuracy')
        worksheet.write('A3', 'Mean class IoU')

        worksheet.write('A5','road          ')
        worksheet.write('A6','sidewalk      ')
        worksheet.write('A7','building      ')
        worksheet.write('A8','wall          ')
        worksheet.write('A9','fence         ')
        worksheet.write('A10','pole          ')
        worksheet.write('A11','traffic light ')
        worksheet.write('A12','traffic sign  ')
        worksheet.write('A13','vegetation    ')
        worksheet.write('A14','terrain       ')
        worksheet.write('A15','sky           ')
        worksheet.write('A16','person        ')
        worksheet.write('A17','rider         ')
        worksheet.write('A18','car           ')
        worksheet.write('A19','truck         ')
        worksheet.write('A20','bus           ')
        worksheet.write('A21','train         ')
        worksheet.write('A22','motorcycle    ')
        worksheet.write('A23','bicycle       ')


        worksheet.write('B5', 'acc')
        worksheet.write('C5', 'iou')

        overalls = 1
        specs = 5
        for line in lines:
            for look in look_for :
                if look in line:
                    if "iou = " in line[len(look):]:
                        print(line[len(look):len(look)+8]) 
                        print(line[len(look)+16:])
                        worksheet.write('B'+str(specs),float(line[len(look):len(look)+8]))
                        worksheet.write('C'+str(specs),float(line[len(look)+16:-2]))
                        specs+=1
                    else:
                        print(line[len(look):]) 
                        worksheet.write('B'+str(overalls), float(line[len(look):-2]))
                        overalls += 1

workbook.close()