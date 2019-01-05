from tkinter import *
from tkinter import ttk
import datetime
import math
from pathlib import Path
import os, shutil
#---------------------------
import keras
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
from keras.models import Sequential
from pathlib import Path
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
from google_images_download import google_images_download
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import theano
from PIL import Image
from numpy import *
# SKLEARN
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
#-----------------------------
import cv2, time
from keras.models import model_from_json
from pathlib import Path
from keras.preprocessing import image
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import messagebox

class MainPage:
    def __init__(self, master):
        self.master = master
        self.frame = tk.Frame(self.master)
        self.titlelbl = tk.Label(self.frame, text = "Easy CNN")
        self.titlelbl.config(font=("Helvetica", 20))
        self.titlelbl.pack()
        self.namelbl = tk.Label(self.frame, text = "made by Bryan Kyritz")
        self.namelbl.config(font=("Helvetica", 10))
        self.namelbl.pack()
        #tensorflow logo
        dir_path = os.path.dirname(os.path.realpath(__file__))
        load = Image.open(dir_path + "\data\pictures\logo.png")

        image = tk.PhotoImage(file=dir_path + "\data\pictures\logo.png")

        self.tfLogoPic = tk.Label(self.frame, image = image)
        self.tfLogoPic.config(image=image)
        self.tfLogoPic.photo = image
        self.tfLogoPic.pack()
        # -----------------------------------------------------------------
        self.titlelbl = tk.Label(self.frame, text = "Menu")
        self.titlelbl.config(font=("Helvetica", 12))
        self.titlelbl.pack()
        self.helpBtn = tk.Button(self.frame, text = 'More Info', width = 25, command = self.help_window)
        self.helpBtn.pack()
        self.trainBtn = tk.Button(self.frame, text = 'Train Model', width = 25, command = self.train_window)
        self.trainBtn.pack()
        self.testBtn = tk.Button(self.frame, text = 'Test Model', width = 25, command = self.test_window)
        self.testBtn.pack()
        self.genCodeBtn = tk.Button(self.frame, text = 'Generate Code', width = 25, command = self.genCode_window)
        self.genCodeBtn.pack()
        self.getDataOnlineBtn = tk.Button(self.frame, text = 'Get Data from online', width = 25, command = self.getDataOnline_window)
        self.getDataOnlineBtn.pack()
        self.getDataFromVideoBtn = tk.Button(self.frame, text = 'Get Data from video', width = 25, command = self.getDataVideo_window)
        self.getDataFromVideoBtn.pack()
        self.analyzeVidBtn = tk.Button(self.frame, text = 'Analyze pre-recorded video', width = 25, command = self.analyzeVid_window)
        self.analyzeVidBtn.pack()
        self.analyzeVidBtn = tk.Button(self.frame, text = 'Analyze image', width = 25, command = self.analyzeImg_window)
        self.analyzeVidBtn.pack()
        self.frame.pack()
    def help_window(self):
        self.newWindow = tk.Toplevel(self.master)
        self.app = HelpPage(self.newWindow)
    def train_window(self):
        self.newWindow = tk.Toplevel(self.master)
        self.app = TrainPage(self.newWindow)
    def test_window(self):
        self.newWindow = tk.Toplevel(self.master)
        self.app = TestPage(self.newWindow)
    def genCode_window(self):
        self.newWindow = tk.Toplevel(self.master)
        self.app = GenCodePage(self.newWindow)
    def getDataOnline_window(self):
        self.newWindow = tk.Toplevel(self.master)
        self.app = GetDataOnlinePage(self.newWindow)
    def getDataVideo_window(self):
        self.newWindow = tk.Toplevel(self.master)
        self.app = GetDataFromVideo(self.newWindow)
    def analyzeVid_window(self):
        self.newWindow = tk.Toplevel(self.master)
        self.app = analyzeVideoPage(self.newWindow)
    def analyzeImg_window(self):
        self.newWindow = tk.Toplevel(self.master)
        self.app = analyzeImagePage(self.newWindow)
class HelpPage:
    def __init__(self, master):
        titleFontSize = 20
        subtitleFontSize = 16
        answerFontSize = 12
        self.master = master
        self.frame = tk.Frame(self.master)
        self.titleMessage = tk.Label(self.master, text="Help")
        self.titleMessage.config(font=("Helvetica", titleFontSize))
        self.titleMessage.pack()

        self.one = tk.Label(self.master,  text="What does this application do?")
        self.one.config(font=("Helvetica", subtitleFontSize))
        self.one.pack()
        self.oneA = tk.Label(self.master, justify=LEFT, text ="    This app was made by Bryan Kyritz in order to make the implication of \n" +
                                                "    Convolutional neural network(CNN) for image detection easy for those who do not know \n" +
                                                "    how to use tensorflow. This app can only detect if a single object is present in an image \n"+
                                                "    or video. Future updates will include the ability to:\n\n"+
                                                "    •Train for multiple objects\n"+
                                                "    •Generate bounding boxes around the objects by implementing RCNN or YOLO\n"+
                                                "    •More customizability (Changing the training settings within the app")
        self.oneA.config(font=("Helvetica", answerFontSize))
        self.oneA.pack()

        self.two = tk.Label(self.master,  text="How does this app work?")
        self.two.config(font=("Helvetica", subtitleFontSize))
        self.two.pack()
        self.twoA = tk.Label(self.master, justify=LEFT, text= "    This app uses Tensorflow, developed by Google, to build a neural netowork which\n"+
                                                    "    is a mathematical model that can learn to detect patterns in order to solve problems.\n"+
                                                    "    The model that is built is a Convolutional Neural Network(CNN) a convolutional neural network \n"+
                                                    "    is a class of deep neural networks, most commonly applied to analyzing visual imagery(computer vision). CNNs use a \n"+
                                                    "    variation of multilayer perceptrons designed to require minimal preprocessing.")
        self.twoA.config(font=("Helvetica", answerFontSize))
        self.twoA.pack()
        self.frame.pack()
class TrainPage:

    nameOfObject = ""
    def __init__(self, master):
        self.master = master
        self.frame = tk.Frame(self.master)
        self.title = tk.Label(self.master, text="Train Model \n")
        self.title.config(font=("Helvetica", 20))
        self.title.pack()
        self.msglbl = tk.Label(self.master, text = "*Make sure you put the data into the training folders*")
        self.msglbl.pack()
        self.nameOfObjectlbl = tk.Label(self.master,text = "Enter the object you are training for:")
        self.nameOfObjectlbl.pack()
        self.nameOfObjTrainingtxt = tk.Text(self.master,width = 20, height = 1)
        self.nameOfObjTrainingtxt.pack()
        self.trainBtn = tk.Button(self.master, text="Train", command = self.train)
        self.trainBtn.pack()
        self.frame.pack()
    #------------------------------------------------------------------------------Deletes all files in a directory
    def deleteFilesInDir(self, folder):
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(e)
    def train(self):
        print("Train Model")

        self.modelName = self.nameOfObjTrainingtxt.get("1.0",END).replace('\n','')
        debug = True
        dir_path = os.path.dirname(os.path.realpath(__file__))
        nameOfObject = self.modelName
        print(dir_path)
        #-----------------------------------------------------------------------Directories being used to get images and save images
        org_Obj_dir = dir_path + r"\training_data\object"
        org_not_Obj_dir = dir_path + r"\training_data\not_object"
        edited_obj_dir = dir_path + r"\training_data\edited_object"
        edited_not_obj_dir = dir_path + r"\training_data\edited_not_object"
        all_data_new = dir_path + r"\training_data\allDataNew"

        #pictures are downsized to 200 by 200 pixels
        img_rows, img_cols = 200, 200

        #number of channels (red, green, blue)
        img_channels = 3

        print("------------------------------------------------------------------")

        self.deleteFilesInDir(all_data_new)
        self.deleteFilesInDir(edited_not_obj_dir)
        self.deleteFilesInDir(edited_obj_dir)
        print("Reset folders")

        listing = os.listdir(org_Obj_dir)
        num_samples_obj = size(listing)
        print("Number of object samples:" + str(num_samples_obj))

        for file in listing:
            im = Image.open(org_Obj_dir + '\\' + file)
            img = im.resize((img_rows, img_cols))
            img.save(edited_obj_dir + '\\' + file, "JPEG")
            img.save(all_data_new + '\\' + file, "JPEG")

        if (debug):
            print("Done editing pictures!")

        # converts images into an array
        imlist = os.listdir(edited_obj_dir)
        im1 = array(Image.open(edited_obj_dir + '\\' + imlist[0]))  # open one image to get size
        m, n = im1.shape[0:2]  # get the size of the images
        imnbr = len(imlist)  # get the number of images

        # -------------------------------------------------------------Edits noncube pictures
        listing = os.listdir(org_not_Obj_dir)
        num_samples_not_obj = size(listing)

        if (debug):
            print("Number of non-object samples:" + str(num_samples_not_obj))

        for file in listing:
            im = Image.open(org_not_Obj_dir + '\\' + file)
            img = im.resize((img_rows, img_cols))
            img.save(edited_not_obj_dir + '\\' + file, "JPEG")
            img.save(all_data_new + '\\' + file, "JPEG")

        if (debug):
            print("Done editing object pictures!")

        # converts images into an array
        imlist2 = os.listdir(edited_not_obj_dir)
        im2 = array(Image.open(edited_not_obj_dir + '\\' + imlist2[0]))  # open one image to get size
        m, n = im2.shape[0:2]  # get the size of the images
        imnbr = len(imlist2)  # get the number of images
        # -------------------------------------------------------------Creates matrix to store all flattened images

        both_lists = imlist + imlist2

        print(len(both_lists))

        try:
            immatrix = array([array(Image.open(all_data_new + '\\' + im3)).flatten()
                          for im3 in both_lists], 'f')  #'f'
        except ValueError:
            print("error in classic spot")
            messagebox.showerror("ERROR", "There is an error with your training data")
            self.deleteFilesInDir(all_data_new)
            self.deleteFilesInDir(edited_not_obj_dir)
            self.deleteFilesInDir(edited_obj_dir)
            return 0
        total_Samples = num_samples_obj + num_samples_not_obj
        label = np.ones((total_Samples,), dtype=int)
        label[0:num_samples_obj] = 1  # object
        label[num_samples_not_obj:] = 0  # not object

        data, Label = shuffle(immatrix, label, random_state=2)
        train_data = [data, Label]

        print(train_data[0].shape)
        print(train_data[1].shape)

        # -------------------------------------------------------------Neural network Settings
        # batch_size to train
        batch_size = 32
        # number of output classes
        nb_classes = 2  # 1 = has cube. 0 = doesnt have cube
        # number of epochs to train
        nb_epoch = 20
        # number of convolutional filters to use
        nb_filters = 32
        # size of pooling area for max pooling
        nb_pool = 2
        # convolution kernel size
        nb_conv = 3
        # -------------------------------------------------------------split X and y into training and testing sets
        (X, y) = (train_data[0], train_data[1])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, img_channels)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, img_channels)

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

        X_train /= 255
        X_test /= 255
        if (debug):
            print('X_train shape:', X_train.shape)
            print(X_train.shape[0], 'train samples')
            print(X_test.shape[0], 'test samples')

        Y_train = np_utils.to_categorical(y_train, nb_classes)
        Y_test = np_utils.to_categorical(y_test, nb_classes)
        # -------------------------------------------------------------Neural Network Model
        model = Sequential()

        model.add(Conv2D(nb_filters, (nb_conv, nb_conv),
                         padding='same',
                         input_shape=(img_rows, img_cols, img_channels)))
        convout1 = Activation('relu')
        model.add(convout1)
        model.add(Conv2D(nb_filters, (nb_conv, nb_conv), padding='same'))

        convout2 = Activation('relu')
        model.add(convout2)
        model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adadelta',  # 'adam' also an option
                      metrics=['accuracy'])
        # -------------------------------------------------------------Train the Neural network
        model.fit(
            X_train,
            Y_train,
            batch_size=batch_size,
            epochs=nb_epoch,
            verbose=1,
            validation_data=(X_test, Y_test), )

        model.fit(
            X_train,
            Y_train,
            batch_size=batch_size,
            epochs=nb_epoch,
            verbose=1,
            validation_split=0.2, )

        # Save neural network's trained weights and structure
        savelink = dir_path + r"\saved_Weights\model_structure_" + nameOfObject + ".json"

        model_structure = model.to_json()
        f = Path(savelink)
        f.write_text(model_structure)

        saveLink2 = dir_path + r"\saved_Weights\model_weights_" + nameOfObject + ".h5"
        model.save_weights(saveLink2)
        self.deleteFilesInDir(all_data_new)
        self.deleteFilesInDir(edited_not_obj_dir)
        self.deleteFilesInDir(edited_obj_dir)
        messagebox.showinfo("Good News", "Done Training!")
        print("Done!")
class TestPage:
    save = False
    def __init__(self, master):
        titleFontSize = 20
        subtitleFontSize = 16
        answerFontSize = 12
        self.master = master
        self.frame = tk.Frame(self.master)
        self.titleMessage = tk.Label(self.master, text="Testing")
        self.titleMessage.config(font=("Helvetica", titleFontSize))
        self.titleMessage.pack()
        self.trainlbl = tk.Label(self.master, text = "Enter the Name of object:")
        self.trainlbl.pack()
        self.nameOfObjTrainingtxt = tk.Text(self.master,width = 20, height = 1)
        self.nameOfObjTrainingtxt.pack()
        self.askSave = tk.Checkbutton(self.master, text = "Save", command = self.check)
        self.askSave.pack()
        self.cameraBtn = tk.Button(self.master, text="Camera", command = self.camera)
        self.cameraBtn.pack()
        self.frame.pack()

    def check(self):
        if self.save == True:
            self.save = False
        else:
            self.save = True
    def camera(self):

        foundPath = True
        nameOfObject = self.nameOfObjTrainingtxt.get("1.0",END).replace('\n','')

        dir_path = os.path.dirname(os.path.realpath(__file__))

        savelink = dir_path + r"\saved_Weights\model_structure_" + nameOfObject + ".json"
        saveLink2 = dir_path + r"\saved_Weights\model_weights_" + nameOfObject + ".h5"

        print(dir_path)

        class_labels = ["Not Object",nameOfObject]
        #Load the json file that contains the model's structure
        try:
            foundPath = True
            f = Path(savelink)
            model_structure = f.read_text()
        except IOError:
            foundPath = False
            messagebox.showerror("ERROR", "Could not find weights for: "+nameOfObject)

        if foundPath:
            f = Path(savelink)
            model_structure = f.read_text()
            # Recreate the Keras model object from the json data
            model = model_from_json(model_structure)

            # Re-load the model's trained weights
            model.load_weights(saveLink2)

            # -----------------------------------------------------------------------------------Resizes image
            img_rows = 200  # what we converted all the training data to (200x200 image)
            img_cols = 200

            #---------------------------------------------------------SAVING VIDEO INFO
            FILE_OUTPUT = nameOfObject + '.avi'

            if os.path.isfile(FILE_OUTPUT): #deletes old video with thw same name
                os.remove(FILE_OUTPUT)

            video = cv2.VideoCapture(0)
            if self.save == True:
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                width = video.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
                height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = video.get(cv2.CAP_PROP_FPS)

                out = cv2.VideoWriter(FILE_OUTPUT, fourcc, 20.0, (int(width), int(height)))

            while True:
                check, frame = video.read()
                cv2.imwrite('currentFrame.jpg', frame)

                im = Image.open("currentFrame.jpg")  # image name
                img = im.resize((img_rows, img_cols))
                image_to_test = image.img_to_array(img)
                list_of_images = np.expand_dims(image_to_test, axis=0)
                results = model.predict(list_of_images)
                single_result = results[0]
                most_likely_class_index = int(np.argmax(single_result))
                class_likelihood = single_result[most_likely_class_index]
                class_label = class_labels[most_likely_class_index]

                # Print the result
                print("This is image is a {} - Likelihood: {:2f}".format(class_label, class_likelihood))
                key = cv2.waitKey(10)
                if key == ord('q'):
                    break

                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (80, 30)
                fontScale = 0.5
                fontColor = (255, 255, 255)
                lineType = 2

                cv2.putText(frame,str(class_label) + "|" + str(class_likelihood),
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            lineType)
                # Edit image in here
                if most_likely_class_index == 0:
                    cv2.circle(frame, (50, 50), 20, (0, 0, 255), -1) #not object indicator
                else:
                    cv2.circle(frame, (50, 50), 20, (0, 255, 0), -1) #object indicator


                # Saving image in here
                if self.save == True:  # saves video
                    out.write(frame)
                #-----------------------
                cv2.imshow("Capturing", frame)
            if self.save == True:
                out.release()
                messagebox.showinfo("Good News", "Saved Video!")
            video.release()
            cv2.destroyAllWindows()
class GenCodePage:
    def __init__(self, master):
        titleFontSize = 20
        subtitleFontSize = 16
        answerFontSize = 12
        self.master = master
        self.frame = tk.Frame(self.master)
        self.titleMessage = tk.Label(self.master, text="Generate Code")
        self.titleMessage.config(font=("Helvetica", titleFontSize))
        self.titleMessage.pack()
        self.trainlbl = tk.Label(self.master, text="Enter the Name of object:")
        self.trainlbl.pack()
        self.nameOfObjTrainingtxt = tk.Text(self.master,width = 20, height = 1)
        self.nameOfObjTrainingtxt.pack()
        self.genBtn = tk.Button(self.master, text="Generate", command = self.genCode)
        self.genBtn.pack()

    def genCode(self):
        nameOfObject = self.nameOfObjTrainingtxt.get("1.0", END).replace('\n', '')
        f = open(nameOfObject + "_Generated_Code" + ".txt", "w")
        f.write("from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation\n"+
                "from keras.models import Sequential\n"+
                "from pathlib import Path\n"+
                "from keras.optimizers import SGD,RMSprop,adam\n"+
                "def camera(self):\n" +
                "import cv2, time\n"+
                "from keras.models import model_from_json\n"+
                "from pathlib import Path\n"+
                "from keras.preprocessing import image\n"+
                "import numpy as np\n"+
                "import theano\n" +
                "\tfoundPath = True\n" +
                "\tdir_path = os.path.dirname(os.path.realpath(__file__))\n" +
                "\tsavelink = dir_path + r\"\saved_Weights\model_structure_\" + nameOfObject + \".json\"\n" +
                "\tsaveLink2 = dir_path + r\"\saved_Weights\model_weights_\" + nameOfObject + \".h5\"\n" +
                "\tclass_labels = [\"Not Object\", nameOfObject,]\n" +
                "\ttry:\n" +
                "\t\tfoundPath = True\n" +
                "\t\tf = Path(savelink)\n" +
                "\t\tmodel_structure = f.read_text()\n" +
                "\texcept IOError:\n" +
                "\t\tfoundPath = False\n" +
                "\t\tmessagebox.showerror(\"ERROR\", \"Could not find weights for: \"+nameOfObject)\n" +
                "\tif foundPath:\n" +
                "\t\tf = Path(savelink)\n" +
                "\t\tfmodel_structure = f.read_text()\n" +
                "\t\tmodel = model_from_json(model_structure)\n" +
                "\t\tmodel.load_weights(saveLink2)\n" +
                "\t\timg_rows = 200\n" +
                "\t\timg_cols = 200\n" +
                "\t\tvideo = cv2.VideoCapture(0)\n" +
                #---------------------------------------------------
                "\t\tim = Image.open(\"currentFrame.jpg\")  # image name\n" +
                "\t\timg = im.resize((img_rows, img_cols))\n" +
                "\t\timage_to_test = image.img_to_array(img)\n" +
                "\t\tlist_of_images = np.expand_dims(image_to_test, axis=0)\n" +
                "\t\tresults = model.predict(list_of_images)\n" +
                "\t\tsingle_result = results[0]\n" +
                "\t\tmost_likely_class_index = int(np.argmax(single_result))\n" +
                "\t\tclass_likelihood = single_result[most_likely_class_index]\n" +
                "\t\tclass_label = class_labels[most_likely_class_index]\n" +
                "\t\tprint(\"This is image is a {} - Likelihood: {:2f}\".format(class_label, class_likelihood))\n" +
                "\t\tif most_likely_class_index == 0:\n" +
                "\t\t\treturn False\n" +
                "\t\telse:\n" +
                "\t\t\treturn True\n"
                )
class GetDataOnlinePage:
    saveChoice = 1 #1 = normal   | #2 = different

    def __init__(self, master):
        titleFontSize = 20
        subtitleFontSize = 16
        answerFontSize = 12
        self.master = master
        self.frame = tk.Frame(self.master)
        self.titleMessage = tk.Label(self.master, text="Get Data Online")
        self.titleMessage.config(font=("Helvetica", titleFontSize))
        self.titleMessage.pack()
        self.trainlbl = tk.Label(self.master, text="Enter the Name of object:")
        self.trainlbl.pack()
        self.nameOfObjTrainingtxt = tk.Text(self.master,width = 20, height = 1)
        self.nameOfObjTrainingtxt.pack()
        self.numlbl = tk.Label(self.master, text="How many Images?")
        self.numlbl.pack()
        self.numtxt = tk.Text(self.master,width = 20, height = 1)
        self.numtxt.pack()
        self.saveNorm = tk.Radiobutton(self.master, text="Normal Save(may overide old data)", variable=self.saveChoice, value=1).pack(anchor=W)
        self.saveDif = tk.Radiobutton(self.master, text="Different save", variable=self.saveChoice, value=2).pack(anchor=W)
        self.dirlbl = tk.Label(self.master, text="Directory to save(only if you selected \"Different save\"")
        self.dirlbl.pack()
        self.dirtxt = tk.Text(self.master,width = 20, height = 1)
        self.dirtxt.pack()
        self.genBtn = tk.Button(self.master, text="Get Data", command = self.getData)
        self.genBtn.pack()
    #---------------------------------------------------------------------------------Delets broken downloads
    def deleteBrokenFiles(self, folder):
        counter = 0
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            ending = file_path[-3:]
            if ending != "jpg":
                counter = counter+1
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
        print("Deleted " + str(counter) + " broken pictures")
    #----------------------------------------------------------------------------------Downloads and organizes saved images from google images, used to build a dataset
    def getData(self):

        nameOfObject = self.nameOfObjTrainingtxt.get("1.0",END).replace('\n','')
        numLimit = self.numtxt.get("1.0",END).replace('\n','')

        isVal = True
        if nameOfObject == "":
            isVal = False
        if numLimit == "":
            isVal = False

        if isVal:
            self.loadingLbl = tk.Label(self.master, text="Downloading...This may take a while...")
            self.loadingLbl.pack()
            nameOfObject = self.nameOfObjTrainingtxt.get("1.0",END).replace('\n','')
            numLimit = self.numtxt.get("1.0",END).replace('\n','')

            dir_path = os.path.dirname(os.path.realpath(__file__))

            response = google_images_download.googleimagesdownload()
            arguments = {"keywords": nameOfObject, "limit": numLimit, "format": "jpg", "chromedriver": dir_path + r"\data\chromedriver.exe"}
            absolute_image_paths = response.download(arguments)

            self.loadingLbl.destroy()
            folder = dir_path + "\downloads\\" + nameOfObject
            self.deleteBrokenFiles(folder)
            messagebox.showinfo("Good News", "Done Downloading!\nIt is recommended to check the pictures.")
        else:
            if nameOfObject == "":
                messagebox.showerror("ERROR", "Enter the name of the object")
            if numLimit == "":
               messagebox.showerror("ERROR", "Enter the number images")
class GetDataFromVideo:
    save = False
    def __init__(self, master):
        titleFontSize = 20
        subtitleFontSize = 16
        answerFontSize = 12
        self.master = master
        self.frame = tk.Frame(self.master)
        self.titleMessage = tk.Label(self.master, text="Generate images from video")
        self.titleMessage.config(font=("Helvetica", titleFontSize))
        self.titleMessage.pack()
        self.pathlbl = tk.Label(self.master, text="Directory of video")
        self.pathlbl.pack()
        self.pathtxt = tk.Text(self.master,width = 20, height = 1)
        self.pathtxt.pack()
        self.askSaveDir = tk.Checkbutton(self.master, text = "Save in Different Location?", command = self.check)
        self.askSaveDir.pack()
        self.savePathlbl = tk.Label(self.master, text="Save Directory of images")
        self.savePathlbl.pack()
        self.savePathtxt = tk.Text(self.master,width = 20, height = 1)
        self.savePathtxt.pack()
        self.genBtn = tk.Button(self.master, text="Generate", command = self.splitVideo)
        self.genBtn.pack()
    def check(self):
        if self.save == True:
            self.save = False
        else:
            self.save = True
    def splitVideo(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))

        path = self.pathtxt.get("1.0", END).replace('\n', '')

        if path == "":
            messagebox.showerror("ERROR", "Enter in the directory of the video.")
            return 0
        if not os.path.exists(path):
            messagebox.showerror("ERROR", "Video Path does not exist")
            return 0
        #checks if folder exists and if it doesnt, then makes one
        if self.save == True:
            if self.savePathtxt.get("1.0", END).replace('\n', '') == "":
                messagebox.showerror("ERROR", "Enter in another save directory or save in the default folder")
                return 0
            else:
                savePath = self.savePathtxt.get("1.0", END).replace('\n', '')
        else:
            try:
                nameOfFile = path
                index = nameOfFile.rfind("\\")

                nameOfFile = nameOfFile[index+1:-4]
                print(nameOfFile)
                if not os.path.exists(dir_path + '\downloads\\' + nameOfFile):
                    os.makedirs(dir_path+ '\downloads\\' + nameOfFile)
                    print("Made folder for image")
                savePath = dir_path + "\downloads\\" + nameOfFile
            except OSError:
                print('Error: Creating directory of data')

        videoFile = path
        imagesFolder = savePath
        currentFrame = 0

        cap = cv2.VideoCapture(path)
        while (True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                break
            # Saves image of the current frame in jpg file
            name = imagesFolder +"\\" + nameOfFile + "_"+str(currentFrame) + '.jpg'
            print('Creating...' + name)
            cv2.imwrite(name, frame)

            # To stop duplicate images
            currentFrame += 1


        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
        messagebox.showinfo("Good News", "Done Saving images! Images saved in: \n"+imagesFolder)
class analyzeVideoPage:
    def __init__(self, master):
        titleFontSize = 20
        subtitleFontSize = 16
        answerFontSize = 12
        self.master = master
        self.frame = tk.Frame(self.master)
        self.titleMessage = tk.Label(self.master, text="Analyze Video")
        self.titleMessage.config(font=("Helvetica", titleFontSize))
        self.titleMessage.pack()
        self.nameOfObjTraininglbl = tk.Label(self.master, text="Name of model")
        self.nameOfObjTraininglbl.pack()
        self.nameOfObjTrainingtxt = tk.Text(self.master,width = 20, height = 1)
        self.nameOfObjTrainingtxt.pack()
        self.pathlbl = tk.Label(self.master, text="Directory of video")
        self.pathlbl.pack()
        self.pathtxt = tk.Text(self.master,width = 20, height = 1)
        self.pathtxt.pack()
        self.genBtn = tk.Button(self.master, text="Generate", command = self.analyze)
        self.genBtn.pack()
    def analyze(self):

        foundPath = True
        nameOfObject = self.nameOfObjTrainingtxt.get("1.0",END).replace('\n','')
        videoPath = self.pathtxt.get("1.0",END).replace('\n','')
        dir_path = os.path.dirname(os.path.realpath(__file__))

        savelink = dir_path + r"\saved_Weights\model_structure_" + nameOfObject + ".json"
        saveLink2 = dir_path + r"\saved_Weights\model_weights_" + nameOfObject + ".h5"

        print(dir_path)

        class_labels = ["Not Object",nameOfObject]
        #Load the json file that contains the model's structure
        try:
            foundPath = True
            f = Path(savelink)
            model_structure = f.read_text()
        except IOError:
            foundPath = False
            messagebox.showerror("ERROR", "Could not find weights for: "+nameOfObject)

        if foundPath:
            f = Path(savelink)
            model_structure = f.read_text()
            # Recreate the Keras model object from the json data
            model = model_from_json(model_structure)

            # Re-load the model's trained weights
            model.load_weights(saveLink2)

            # -----------------------------------------------------------------------------------Resizes image
            img_rows = 200  # what we converted all the training data to (200x200 image)
            img_cols = 200

            #---------------------------------------------------------SAVING VIDEO INFO
            FILE_OUTPUT = nameOfObject + '.avi'

            if os.path.isfile(FILE_OUTPUT): #deletes old video with thw same name
                os.remove(FILE_OUTPUT)

            video = cv2.VideoCapture(videoPath)

            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            width = video.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
            height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = video.get(cv2.CAP_PROP_FPS)

            out = cv2.VideoWriter(FILE_OUTPUT, fourcc, 20.0, (int(width), int(height)))

            while True:
                check, frame = video.read()
                cv2.imwrite('currentFrame.jpg', frame)

                if check:
                    try:
                        im = Image.open("currentFrame.jpg")  # image name
                    except IOError:
                        print("Error")
                        messagebox.showerror("ERROR", "An error ocurred!")
                else:
                    messagebox.showinfo("Good News", "Saved Analyzed Video!")
                    return 0
                img = im.resize((img_rows, img_cols))
                image_to_test = image.img_to_array(img)
                list_of_images = np.expand_dims(image_to_test, axis=0)
                results = model.predict(list_of_images)
                single_result = results[0]
                most_likely_class_index = int(np.argmax(single_result))
                class_likelihood = single_result[most_likely_class_index]
                class_label = class_labels[most_likely_class_index]

                # Print the result
                print("This is image is a {} - Likelihood: {:2f}".format(class_label, class_likelihood))
                key = cv2.waitKey(10)
                if key == ord('q'):
                    break

                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (80, 50)
                fontScale = 0.5
                fontColor = (255, 255, 255)
                lineType = 2

                cv2.putText(frame,str(class_label) + "|" + str(class_likelihood),
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            lineType)
                # Edit image in here
                if most_likely_class_index == 0:
                    cv2.circle(frame, (50, 50), 20, (0, 0, 255), -1) #not object indicator
                else:
                    cv2.circle(frame, (50, 50), 20, (0, 255, 0), -1) #object indicator

                # Saving image in here  # saves video
                out.write(frame)
                #-----------------------
                cv2.imshow("Analyzing", frame)
            out.release()
            messagebox.showinfo("Good News", "Saved Analyzed Video!")
            video.release()
            cv2.destroyAllWindows()
class analyzeImagePage:
    def __init__(self, master):
        titleFontSize = 20
        subtitleFontSize = 16
        answerFontSize = 12
        self.master = master
        self.frame = tk.Frame(self.master)
        self.titleMessage = tk.Label(self.master, text="Analyze Image")
        self.titleMessage.config(font=("Helvetica", titleFontSize))
        self.titleMessage.pack()
        self.nameOfObjTraininglbl = tk.Label(self.master, text="Name of model")
        self.nameOfObjTraininglbl.pack()
        self.nameOfObjTrainingtxt = tk.Text(self.master,width = 20, height = 1)
        self.nameOfObjTrainingtxt.pack()
        self.pathlbl = tk.Label(self.master, text="Directory of image")
        self.pathlbl.pack()
        self.pathtxt = tk.Text(self.master,width = 20, height = 1)
        self.pathtxt.pack()
        self.genBtn = tk.Button(self.master, text="Analyze", command = self.analyze)
        self.genBtn.pack()

    def analyze(self):
        foundPath = True
        nameOfObject = self.nameOfObjTrainingtxt.get("1.0",END).replace('\n','')
        imgPath = self.pathtxt.get("1.0",END).replace('\n','')
        dir_path = os.path.dirname(os.path.realpath(__file__))

        savelink = dir_path + r"\saved_Weights\model_structure_" + nameOfObject + ".json"
        saveLink2 = dir_path + r"\saved_Weights\model_weights_" + nameOfObject + ".h5"

        print(dir_path)

        class_labels = ["Not Object",nameOfObject]
        #Load the json file that contains the model's structure
        try:
            foundPath = True
            f = Path(savelink)
            model_structure = f.read_text()
        except IOError:
            foundPath = False
            messagebox.showerror("ERROR", "Could not find weights for: "+nameOfObject)

        if foundPath:
            f = Path(savelink)
            model_structure = f.read_text()
            # Recreate the Keras model object from the json data
            model = model_from_json(model_structure)

            # Re-load the model's trained weights
            model.load_weights(saveLink2)

            # -----------------------------------------------------------------------------------Resizes image
            img_rows = 200  # what we converted all the training data to (200x200 image)
            img_cols = 200

            #---------------------------------------------------------SAVING Image INFO
            try:
                im = Image.open(imgPath)  # image name
            except IOError:
                print("Error")
                messagebox.showerror("ERROR", "Image not found!")

            img = im.resize((img_rows, img_cols))
            image_to_test = image.img_to_array(img)
            list_of_images = np.expand_dims(image_to_test, axis=0)
            results = model.predict(list_of_images)
            single_result = results[0]
            most_likely_class_index = int(np.argmax(single_result))
            class_likelihood = single_result[most_likely_class_index]
            class_label = class_labels[most_likely_class_index]

            # Print the result
            print("This is image is a {} - Likelihood: {:2f}".format(class_label, class_likelihood))

            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (80, 50)
            fontScale = 0.5
            fontColor = (255, 255, 255)
            lineType = 2

            cv2.putText(img,str(class_label) + "|" + str(class_likelihood),
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)
            # Edit image in here
            if most_likely_class_index == 0:
                cv2.circle(img, (50, 50), 20, (0, 0, 255), -1) #not object indicator
            else:
                cv2.circle(img, (50, 50), 20, (0, 255, 0), -1) #object indicator

            # Saving image in here  # saves video
            Image.write(img)
            messagebox.showinfo("Good News", "Saved Analyzed Picture!")
def main():
    root = tk.Tk()
    root.minsize(300,300)
    root.title("Easy CNN 1.0")
    dir_path = os.path.dirname(os.path.realpath(__file__))
    root.iconbitmap(dir_path + "\data\pictures\icon.ico")
    root.resizable(False, False)
    center(root)
    app = MainPage(root)
    root.mainloop()
def center(toplevel): #centers the main window
    toplevel.update_idletasks()

    # PyQt way to find the screen resolution
    screen_width = toplevel.winfo_screenwidth()
    screen_height = toplevel.winfo_screenheight()

    x = screen_width/2 - toplevel.winfo_width()/2
    y = screen_height/2 - toplevel.winfo_height()/2
    toplevel.geometry("+%d+%d" % (x, y))

if __name__ == '__main__':
    main()