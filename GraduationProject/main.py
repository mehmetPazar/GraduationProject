'''
Ana program
@Author: Pazar

Çalıştırmak için:
main.py

Yeni kayıt yapmak için:
main.py --mode "input"

'''

import cv2 #opencv kütüphanesi 
from firebase import firebase # realtime veritabanı için firebase
from align_custom import AlignCustom  #align_custom.py'den AlignCustom classını import ettik
from face_feature import FaceFeature #face_feature.py'den FaceFeature classını import ettik
from mtcnn_detect import MTCNNDetect #mtcnn_detect.py'den MTCNNDetect classını import ettik
from tf_graph import FaceRecGraph #tf_graph.py'den FaceRecGraph classını import ettik
import argparse
import sys
import json
import time
import numpy as np #gelişmiş matematiksel işlemleri yapmak için numpy
import locale
from datetime import datetime
import datetime
import os

TIMEOUT = 10 #10 seconds
firebase = firebase.FirebaseApplication('https://bitirme-eab1b.firebaseio.com/') #firebase bağlantısı için gereken string
firebaseperson = ["Unknown"]#her framede bulunun yeni kişinin eklendiği dizi
a=0

def main(args): #çalıştırma tipi 
    mode = args.mode
    if(mode == "camera"): #main.py ile çalıştırmak için
        camera_recog()
    elif mode == "input": #main.py --mode "input" ile çalıştırmak için
        create_manual_data();
    else:
        raise ValueError("Unimplemented mode") #hata

def camera_recog(): # yüzü bulması için gereken fonksiyon
    print("[INFO] camera sensor warming up...") 
    vs = cv2.VideoCapture(0); #cameradan giriş al
    detect_time = time.time()
    global a
    while True: #camera çalışır frame akar 
        _,frame = vs.read(); #frame okunur
        rects, landmarks = face_detect.detect_face(frame,60);#yüz boyutu 80*80
        aligns = [] 
        positions = []
        
        for (i, rect) in enumerate(rects):
            aligned_face, face_pos = aligner.align(160,frame,landmarks[:,i])
            if len(aligned_face) == 160 and len(aligned_face[0]) == 160:
                aligns.append(aligned_face) #yüz bulundu ve diziye atıldı
                positions.append(face_pos) #yüzün pozisyonu diziye atıldı
            else: 
                print("Align face failed") #log        
        if(len(aligns) > 0): #bulunan yüz sayısı 0'dan büyük ise 
            features_arr = extract_feature.get_features(aligns)  
            recog_data = findPeople(features_arr,positions) # yüzün pozisyonu ile kim olduğu bulunuyor
            for (i,rect) in enumerate(rects): #framede bulunan yüz sayısı kadar
                cv2.rectangle(frame,(rect[0],rect[1]),(rect[2],rect[3]),(255,0,0)) #yüzü kutu içine alıyoruz
                cv2.putText(frame,recog_data[i][0]+" - "+str(recog_data[i][1])+"%",(rect[0],rect[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv2.LINE_AA)#kutuya kişi adını ve benzeme yüzdesini yazdırıyoruz
                find = firebaseperson.count(recog_data[i][0]) #her frame bulunan ve dizide olmayan kişileri diziye atmak için framede bulunan kişilerle karşılaştırıyoruz
                if(find == 0): #dizide yok ise 
                    firebaseperson.append(recog_data[i][0]) #diziye atıyoruz
                else:
                    continue
                  
        cv2.imshow("Frame",frame) #frame göserilir
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"): #q'ya basılınca programdan çıkma işlemi başlıyor ve bir txt dosyasına yazma işlemi yapılıyor
            read = []
            an = datetime.datetime.now()
            roll_call = open("C:\\tensorflow1\\models\\FaceRec-master\\roll_call.txt", "r+") #roll_call adındaki txt dosyası açılır bu ders başına alınan iki yokamanın sağlaması için yapılır ve firebase kaydedilir.2. defa çalıştırıldığında
            lines=roll_call.readlines() #içindeki her satır okunur (olup olmaması farketmez)
            
            if(len(lines)>0 and len(lines)<=len(firebaseperson)): #satır sayısı 0'dan fazla ise ve o an alınan framedeki kişilerin sayısı satır sayısından fazla ise
                i=0
                while i < len(firebaseperson): 
                    if(len(lines)>i):
                        find=firebaseperson.count(lines[i].strip()) #satırı o anki framede bulunan kişiler ile karşılaştırıyoruz         
                        if(find == 1): #bulunursa
                            result = firebase.post('TarihSaaat/'+datetime.datetime.strftime(an, '%d %B %Y %H:%M')+'/', {i:lines[i].strip()}) #firebase post ediyoruz
                    else: #i satır sayısından fazla olunca döngüden çıkar
                        break
                    i += 1
                roll_call.truncate(0) #txt'deki tüm satırları siler
                    
            elif(len(lines)>0 and len(lines) > len(firebaseperson)):  #satır sayısı 0'dan fazla ise ve o an alınan framedeki kişilerin sayısı satır sayısından az ise
                j=0
                while j < len(firebaseperson):
                    find=firebaseperson.count(lines[j].strip()) #satırı o anki framede bulunan kişiler ile karşılaştırıyoruz        
                    if(find == 1): #bulunursa
                        result = firebase.post('TarihSaaat/'+datetime.datetime.strftime(an, '%d %B %Y %H:%M')+'/', {j:lines[j].strip()}) #firebase post ediyoruz
                    j += 1
                roll_call.truncate(0)#txt'deki tüm satırları siler
            else: #dosya boş ise framede bulunan her kişiyi bir txt dosyasına kaydeder ve ikinci kez çalıştırılana kadar saklar 
                k=0
                while k < len(firebaseperson):
                    roll_call.write(firebaseperson[k]+"\n") #dosyaya yaz
                    k += 1
                    
            roll_call.close() #dosyayı kapat
            break #programdan çık

def findPeople(features_arr, positions, thres = 0.6, percent_thres = 85): #yüzün kime ait olduğunu bulmak için gerekn fonksiyon
    
    f = open('./facerec_128D.txt','r') #facerec_128D adlı txtden okuma işlemi için dosyayı açma izni yüzlerin bulunduğu dosya
    data_set = json.loads(f.read()); #json ile oku
    returnRes = []; 
    for (i,features_128D) in enumerate(features_arr):
        result = "Unknown";
        smallest = sys.maxsize
        for person in data_set.keys(): #kişiler for ile taranır
            person_data = data_set[person][positions[i]]; #kişinin yüz pozisyonları bir matrise atılır
            for data in person_data:
                distance = np.sqrt(np.sum(np.square(data-features_128D)))
                if(distance < smallest):
                    smallest = distance;
                    result = person; #karşılaştırmalar sonucunda kişi bulunur
        percentage =  min(100, 100 * thres / smallest) #yüzdesel olarak ne kadar benzediği yazılır
        if percentage <= percent_thres : #eşik değerinin altındaysa kişi bilinemez
            result = "Unknown" 
        returnRes.append((result,percentage))
    return returnRes #bulunan kişi döndürülür 


def create_manual_data():
    vs = cv2.VideoCapture(0); #kameradan giriş al
    print("Please input new user ID:")
    new_name = input(); #yeni kişinin adı kamera ile alınır
    f = open('./facerec_128D.txt','r'); #facerec_128D.txt okuma işlemi için açılır
    data_set = json.loads(f.read()); #json ile okuma yapılır
    person_imgs = {"Left" : [], "Right": [], "Center": []};
    person_features = {"Left" : [], "Right": [], "Center": []};
    print("Please start turning slowly. Press 'q' to save and add this new user to the dataset");
    while True:
        _, frame = vs.read();
        rects, landmarks = face_detect.detect_face(frame, 80);  
        for (i, rect) in enumerate(rects):
            aligned_frame, pos = aligner.align(160,frame,landmarks[:,i]);
            if len(aligned_frame) == 160 and len(aligned_frame[0]) == 160:
                person_imgs[pos].append(aligned_frame) #yakalanan yüz pozisyonları diziye eklenir sağ sol önden
                cv2.imshow("Captured face", aligned_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    for pos in person_imgs: #yeni kişi bir diziye atılır ,atılmadan önce bazı işlemler yapılır
        person_features[pos] = [np.mean(extract_feature.get_features(person_imgs[pos]),axis=0).tolist()]
    data_set[new_name] = person_features;
    f = open('./facerec_128D.txt', 'w');
    f.write(json.dumps(data_set)) #diziye atılan kişi daha sonra yüzme bulma işleminde kullanılmak üzere dosyaya yazdırılır



if __name__ == '__main__': #main fonksiyon
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, help="Run camera recognition", default="camera") #program çalışırken mode okundu ona göre işlem yapılıcak
    args = parser.parse_args(sys.argv[1:]);
    FRGraph = FaceRecGraph(); #tüm .py'ler tanımlandı
    MTCNNGraph = FaceRecGraph();
    aligner = AlignCustom();
    extract_feature = FaceFeature(FRGraph)
    face_detect = MTCNNDetect(MTCNNGraph, scale_factor=2); #scale_factor daha hızlı algılama için görüntüyü yeniden ölçeklendirir
    main(args);
