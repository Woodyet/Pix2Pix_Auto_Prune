import os
import subprocess
base_dir = "/mnt/f/Data4NN/Pix2Pix/202208311633Flip"
Ground_Truths = base_dir + "/" +"Ground_Truths"
Inital_models = base_dir+ "/" +"Inital_models"
Inital_train = base_dir+ "/" +"Inital_train"
Post_pruned_models = base_dir+ "/" +"Post_pruned_models"
Pre_pruned_models = base_dir+ "/" +"Pre_pruned_models"

all_dirs = [Ground_Truths,Inital_models,Inital_train,Post_pruned_models,Pre_pruned_models]

def Locate_Dirs_in_Dirs(directory_in_str):
		directory = os.fsencode(directory_in_str)
		files = []
		for file in os.listdir(directory):
			filename = os.fsdecode(file)
			#print(os.path.join(directory, file))
			files.append(filename)
		return files

for dir in all_dirs:
    Other_Dirs = Locate_Dirs_in_Dirs(dir)
    for end_dir in Other_Dirs:
        print("python3 ./scripts/eval_cityscapes/evaluate.py --cityscapes_dir /mnt/f/Data4NN/Pix2Pix/ --result_dir " + dir +"/"+end_dir + " --output_dir " +dir +"/"+end_dir)
        process = subprocess.Popen("python3 ./scripts/eval_cityscapes/evaluate.py --cityscapes_dir /mnt/f/Data4NN/Pix2Pix/ --result_dir " + dir +"/"+end_dir + " --output_dir " +dir +"/"+end_dir +" >> output.txt", shell=True, stdout=subprocess.PIPE)
        process.wait()