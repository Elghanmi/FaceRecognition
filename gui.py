import tkinter as tk
from tkinter import filedialog
import imutils
from tkinter import *
from PIL import ImageTk, Image
from tensorflow.keras.preprocessing import image
import numpy as np
import numpy
import cv2
import operator
marge=70
#load the trained model to classify sign
from tensorflow.keras.models import load_model
model = load_model('modele.h5')
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
profile_cascade=cv2.CascadeClassifier("haarcascade_profileface.xml")
#dictionary to label all traffic signs class.
classes= ['pins_Adriana Lima', 'pins_Alex Lawther', 'pins_Alexandra Daddario', 'pins_Alvaro Morte', 'pins_alycia dabnem carey', 'pins_Amanda Crew', 'pins_amber heard', 'pins_Andy Samberg', 'pins_Anne Hathaway', 'pins_Anthony Mackie', 'pins_Avril Lavigne', 'pins_barack obama', 'pins_barbara palvin', 'pins_Ben Affleck', 'pins_Bill Gates', 'pins_Bobby Morley', 'pins_Brenton Thwaites', 'pins_Brian J. Smith', 'pins_Brie Larson', 'pins_camila mendes', 'pins_Chris Evans', 'pins_Chris Hemsworth', 'pins_Chris Pratt', 'pins_Christian Bale', 'pins_Cristiano Ronaldo', 'pins_Danielle Panabaker', 'pins_Dominic Purcell', 'pins_Dwayne Johnson', 'pins_Eliza Taylor', 'pins_Elizabeth Lail', 'pins_elizabeth olsen', 'pins_ellen page', 'pins_elon musk', 'pins_Emilia Clarke', 'pins_Emma Stone', 'pins_Emma Watson', 'pins_gal gadot', 'pins_grant gustin', 'pins_Gwyneth Paltrow', 'pins_Henry Cavil', 'pins_Hugh Jackman', 'pins_Inbar Lavi', 'pins_Irina Shayk', 'pins_Jake Mcdorman', 'pins_Jason Momoa', 'pins_jeff bezos', 'pins_Jennifer Lawrence', 'pins_Jeremy Renner', 'pins_Jessica Barden', 'pins_Jimmy Fallon', 'pins_Johnny Depp', 'pins_Josh Radnor', 'pins_Katharine Mcphee', 'pins_Katherine Langford', 'pins_Keanu Reeves', 'pins_kiernen shipka', 'pins_Krysten Ritter', 'pins_Leonardo DiCaprio', 'pins_Lili Reinhart', 'pins_Lindsey Morgan', 'pins_Lionel Messi', 'pins_Logan Lerman', 'pins_Madelaine Petsch', 'pins_Maisie Williams', 'pins_margot robbie', 'pins_Maria Pedraza', 'pins_Marie Avgeropoulos', 'pins_Mark Ruffalo', 'pins_Mark Zuckerberg', 'pins_Megan Fox', 'pins_melissa fumero', 'pins_Miley Cyrus', 'pins_Millie Bobby Brown', 'pins_Morena Baccarin', 'pins_Morgan Freeman', 'pins_Nadia Hilker', 'pins_Natalie Dormer', 'pins_Natalie Portman', 'pins_Neil Patrick Harris', 'pins_Pedro Alonso', 'pins_Penn Badgley', 'pins_Rami Malek', 'pins_Rebecca Ferguson', 'pins_Richard Harmon', 'pins_Rihanna', 'pins_Robert De Niro', 'pins_Robert Downey Jr', 'pins_Sarah Wayne Callies', 'pins_scarlett johansson', 'pins_Selena Gomez', 'pins_Shakira Isabel Mebarak', 'pins_Sophie Turner', 'pins_Stephen Amell', 'pins_Taylor Swift', 'pins_Tom Cruise', 'pins_tom ellis', 'pins_Tom Hardy', 'pins_Tom Hiddleston', 'pins_Tom Holland', 'pins_Tuppence Middleton', 'pins_Ursula Corbero', 'pins_Wentworth Miller', 'pins_Zac Efron', 'pins_Zendaya', 'pins_Zoe Saldana']

                 
#initialise GUI
top=tk.Tk()
top.geometry('800x600')
top.title('Face Recognition')
top.configure(background='#CDCDCD')

label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)

def classify(file_path):
    global label_packed
    img = image.load_img(file_path, target_size=(160, 160))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255.
    classe = model.predict(x)
    result = np.squeeze(classe)
    result_indices = np.argmax(result)
    sign = "{}    ||   {:.2f}%".format(classes[result_indices],result[result_indices]*100)
    print(sign)
    label.configure(foreground='#011638', text=sign) 
classify_b1 =Button(top)
classify_b =Button(top)
def show_classify_button1():
    global classify_b1
    classify_b.place_forget()
    classify_b1=Button(top,text="Stop Record",command=stopp,padx=10,pady=5)
    classify_b1.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    classify_b1.place(relx=0.79,rely=0.65)
def show_classify_button(file_path):
    global classify_b
    classify_b=Button(top,text="Classify Image",command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)
def stopp():
    global cap
    sign_image.place_forget()
    cap.release()
def upload_image():
    global i
    i=0
    classify_b1.destroy()

    try:

        file_path=filedialog.askopenfilename()
        print(file_path)
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

cap = None
i=0
def upload_video():

        global i
        global cap
        if i ==0 :
            cap = cv2.VideoCapture(0)
            show_classify_button1()
            classify_b.destroy()
        width = int(cap.get(3))
        ref, ca = cap.read()
        if ref ==True:
            ##############################
            ##############################
            ##############################
            aaa=cv2.resize(ca,(160,160))
            x = image.img_to_array(aaa)
            x = np.expand_dims(x, axis=0)
            x /= 255.
            classe = model.predict(x)
            result = np.squeeze(classe)
            result_indices = np.argmax(result)
            sign = "{} *** {:.2f}%".format(classes[result_indices], result[result_indices] * 100)
            print(sign)
            label.configure(foreground='#011638', text=sign)

            ##############################
            ##############################
            tab_face = []
            gray = cv2.cvtColor(ca, cv2.COLOR_BGR2GRAY)
            face = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(5, 5))
            for x, y, w, h in face:
                tab_face.append([x, y, x + w, y + h])
            face = profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            for x, y, w, h in face:
                tab_face.append([x, y, x + w, y + h])
            gray2 = cv2.flip(gray, 1)
            face = profile_cascade.detectMultiScale(gray2, scaleFactor=1.1, minNeighbors=4)
            for x, y, w, h in face:
                tab_face.append([width - x, y, width - (x + w), y + h])
            tab_face = sorted(tab_face, key=operator.itemgetter(0, 1))
            index = 0
            for x, y, x2, y2 in tab_face:
                if not index or (x - tab_face[index - 1][0] > marge or y - tab_face[index - 1][1] > marge):
                    cv2.rectangle(ca, (x, y), (x2, y2), (0, 0, 255), 2)
                index += 1

            cv2.putText(ca, sign, (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)


            ca = imutils.resize(ca ,width=160)
            ca = cv2.resize(ca,(160,160))
            ca = cv2.cvtColor(ca,cv2.COLOR_BGR2RGB)
            im = Image.fromarray(ca)
            imgg = ImageTk.PhotoImage(image=im)

            sign_image.configure(image=imgg)
            sign_image.image = imgg
            i=1
            sign_image.after(1,upload_video)



upload=Button(top,text="Upload an image",command=upload_image,padx=10,pady=5)
upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
upload.place(x=250,y=550)
#*********
#*********
upload1=Button(top,text="Upload an video",command=upload_video,padx=10,pady=5)
upload1.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
upload1.place(x=450,y=550)

#*********
#*********
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Know who you look like",pady=20, font=('arial',20,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
top.mainloop()