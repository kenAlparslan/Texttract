import os
import shutil
dirs = ['advertisement', 'budget', 'email', 'file_folder', 'form', 'handwritten', 'invoice', 'letter', 'memo', 'news_article', 'presentation', 'questionnaire', 'resume', 'scientific_publication', 'scientific_report', 'specification']
def create_train_test_folder(path):
    cut_off = 250
    image_filenames = []
    for d in dirs:
        current_dir = os.path.join(path, d)
        destination = os.path.abspath(path+"/../val/"+ d)
        print(destination)
        files = os.listdir(current_dir)
        test_files = files[:cut_off]
        for f in test_files:
            shutil.move(os.path.join(current_dir,f),destination)        
    return 1
def create_class_folders(path):
    current_dir = os.path.join(path)
    os.chdir(current_dir)
    for d in dirs:
        os.mkdir(d)
    return 1
# Uncomment these two lines to create a test folder out of a train folder.
# print(create_class_folders(os.path.join("rvlcdip-dataset","val")))
print(create_train_test_folder(os.path.join("rvlcdip-dataset","train")))