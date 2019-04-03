import numpy as np
import matplotlib.pyplot as plt
from scipy import misc as mc
def m1(img,rog):
    m, n=img.shape
    maska=np.zeros((m,n))
    m1=np.ones((rog,rog))
    m2=np.zeros((m-2*rog,rog))
    maska=np.concatenate((m1,m2), axis=0)
    maska=np.concatenate((maska, m1), axis=0)
    F=maska.copy()
    m3=np.zeros((m,n-2*rog))
    maska=np.concatenate((maska, m3), axis=1)
    maska=np.concatenate((maska, F), axis=1)
    
    E=np.ones((m/2,n/2))
    E[150:160,:]=0
    E[:,120:128]=0
    E=np.concatenate((E,E[::-1]), axis=0)
    E=np.concatenate((E,np.fliplr(E)), axis=1)
    
    
    return E


#
def m2(img,rog):
    m, n=img.shape
    maska=np.ones((m,n))
    m1=np.zeros((rog,rog))
    m2=np.ones((m-2*rog,rog))
    maska=np.concatenate((m1,m2), axis=0)
    maska=np.concatenate((maska, m1), axis=0)
    F=maska.copy()
    m3=np.ones((m,n-2*rog))
    maska=np.concatenate((maska, m3), axis=1)
    maska=np.concatenate((maska, F), axis=1)
    return maska


#
def m3(img,max1,rog):
    m, n=img.shape
    maska=np.zeros((m/2,n/2))
    maska[1:rog,1:rog]=0
    maska[1:max1,rog:max1]=1
    maska[rog:max1,1:max1]=1
    maska=np.concatenate((maska,np.fliplr(maska)),axis=1)
    maska=np.concatenate((maska,np.flipud(maska)),axis=0)
    return maska


#
def m4(img,max1,rog):
    m, n=img.shape
    maska=np.ones((m/2,n/2))
    maska[1:rog,1:rog]=1
    maska[1:max1,rog:max1]=0
    maska[rog:max1,1:max1]=0
    maska=np.concatenate((maska,np.fliplr(maska)),axis=1)
    maska=np.concatenate((maska,np.flipud(maska)),axis=0)
    return maska


#
#funkcja do sprawdzania czy jest w przedziale
def czy(img):
        M, N = img.shape
        for i in range(M) :
                for j in range (N):
                        if img[i,j]>255:
                                img[i,j]=255
                        if img[i,j]<0:
                                img[i,j]=0
        return img


#
#funkcja wyliczajaca jasnosc i kontrast
def jasnosc(img):
	M, N = img.shape
	j=np.sum(img)/(M*N)
	k=np.sum(np.power(img-j,2))
	k=k/(M*N)
	k=np.power(k,0.5)
	return j,k


#
#wczytanie obrazu
img=mc.imread('kierowca.png')
#przygotowanie obrazu
#zamiana na szarosc
r=img[:,:,0]
g=img[:,:,1]
b=img[:,:,2]
y=0.2126*r+0.7152*g+0.0722*b
y0=y.copy().astype('float64')
print ('\nJasnosc i kontrast obrazu oryginalnego:')
print jasnosc (y/255)
plt.subplot(1,7,1)
plt.imshow(y,cmap=plt.cm.gray ,vmin =0,vmax=255)
#liczenie widma
y=y.astype('float64')
y=np.fft.fft2(y)
y=np.fft.fftshift(y)
plt.subplot(1,7,2)
plt.imshow(np.absolute(np.real(y)),cmap=plt.cm.gray ,vmin =0,vmax=255)
#dodawanie stalej
y=np.fft.ifftshift(y)
y1=y.copy()
y1[0,0]=y1[0,0]+6000000
y1=np.fft.ifft2(y1)
y1=np.real(y1)/255
plt.subplot(1,7,3)
plt.imshow(np.absolute(y1),cmap=plt.cm.gray ,vmin =0,vmax=1)
print ('\nJasnosc i kontrast obrazu po dodaniu stalej:')
print jasnosc (np.absolute(y1))
#mnozeniestalej
y2=y.copy()
y2[0,0]=y2[0,0]*2
y2=np.fft.ifft2(y2)
y2=np.real(y2)/255
#y2=czy(y2)
plt.subplot(1,7,4)
plt.imshow(np.absolute(y2),cmap=plt.cm.gray ,vmin =0,vmax=1)
print ('\nJasnosc i kontrast obrazu po mnozeniu przez stala:')
print jasnosc (np.absolute(y2))
#potegowanie
y3=y.copy()
y3[0,0]=np.power(y3[0,0],2)
y3=np.fft.ifft2(y3)
y3=np.real(y3)/255
#y2=czy(y2)
plt.subplot(1,7,5)
plt.imshow(np.absolute(y3),cmap=plt.cm.gray ,vmin =0,vmax=1)
print ('\nJasnosc i kontrast obrazu po potegowaniu:')
print jasnosc (np.absolute(y3))
#pierwiastkowanie
y4=y.copy()
y4[0,0]=np.power(y4[0,0],0.5)
y4=np.fft.ifft2(y4)
y4=np.real(y4)/255
#y2=czy(y2)
plt.subplot(1,7,6)
plt.imshow(np.absolute(y4),cmap=plt.cm.gray ,vmin =0,vmax=1)
print ('\nJasnosc i kontrast obrazu po pierwiastkowaniu:')
print jasnosc (np.absolute(y4))
#logarytm
y5=y.copy()
x2=5
x2=np.log(x2)
y5[0,0]=np.log(y5[0,0])
y5=np.fft.ifft2(y5)
y5=np.real(y5)/255
plt.subplot(1,7,7)
plt.imshow(np.absolute(y5),cmap=plt.cm.gray ,vmin =0,vmax=1)
print ('\nJasnosc i kontrast obrazu po logarytmowaniu:')
print jasnosc (np.absolute(y5))
plt.show()
############################
#operacje na maskach
#wyswietlenie
plt.subplot(4,5,1)
plt.imshow(img,cmap=plt.cm.gray)
plt.subplot(4,5,6)
plt.imshow(img,cmap=plt.cm.gray)
plt.subplot(4,5,11)
plt.imshow(img,cmap=plt.cm.gray)
plt.subplot(4,5,16)
plt.imshow(img,cmap=plt.cm.gray)
#skalowanie
img_s=y0.copy()
plt.subplot(4,5,2)
plt.imshow(img_s,cmap=plt.cm.gray)
plt.subplot(4,5,7)
plt.imshow(img_s,cmap=plt.cm.gray)
plt.subplot(4,5,12)
plt.imshow(img_s,cmap=plt.cm.gray)
plt.subplot(4,5,17)
plt.imshow(img_s,cmap=plt.cm.gray)
#wyliczenie widma
#transformata furiera - z rzeczywistej do czestotliwosci
img1=np.fft.fft2(img_s.copy())
img2=np.fft.fft2(img_s.copy())
img3=np.fft.fft2(img_s.copy())
img4=np.fft.fft2(img_s.copy())
#przesuniecie
img1=np.fft.fftshift(img1)
img2=np.fft.fftshift(img2)
img3=np.fft.fftshift(img3)
img4=np.fft.fftshift(img4)
#wyswietlenie
plt.subplot(4,5,3)
plt.imshow(np.absolute(np.real(img1)),cmap=plt.cm.gray,vmin=0,vmax=255)
plt.subplot(4,5,8)
plt.imshow(np.absolute(np.real(img1)),cmap=plt.cm.gray,vmin=0,vmax=255)
plt.subplot(4,5,13)
plt.imshow(np.absolute(np.real(img1)),cmap=plt.cm.gray,vmin=0,vmax=255)
plt.subplot(4,5,18)
plt.imshow(np.absolute(np.real(img1)),cmap=plt.cm.gray,vmin=0,vmax=255)
#modyfikacje widma
#utworzenie maski
maska1=m1(img_s.copy(),120)
maska2=m2(img_s.copy(),120)
maska3=m3(img_s.copy(),120,50)
maska4=m4(img_s.copy(),120,50)
plt.subplot(4,5,4)
plt.imshow(maska1,cmap=plt.cm.gray)
plt.subplot(4,5,9)
plt.imshow(maska2,cmap=plt.cm.gray)
plt.subplot(4,5,14)
plt.imshow(maska3,cmap=plt.cm.gray)
plt.subplot(4,5,19)
plt.imshow(maska4,cmap=plt.cm.gray)
#wymnazanie elementow
#img1=np.fft.ifftshift(img1)
img1=np.multiply(maska1,img1)
img2=np.multiply(maska2,img2)
img3=np.multiply(maska3,img3)
img4=np.multiply(maska4,img4)
#zmiana z czestotliwosci
img1 = np.fft.ifft2(img1)
img1 = np.real(img1)
img2 = np.fft.ifft2(img2)
img2 = np.real(img2)
img3 = np.fft.ifft2(img3)
img3 = np.real(img3)
img4 = np.fft.ifft2(img4)
img4 = np.real(img4)
#wyswietlanie
plt.subplot(4,5,5)
plt.imshow(np.absolute(img1),cmap=plt.cm.gray,vmin=0,vmax=255)
plt.subplot(4,5,10)
plt.imshow(np.absolute(img2),cmap=plt.cm.gray,vmin=0,vmax=255)
plt.subplot(4,5,15)
plt.imshow(np.absolute(img3),cmap=plt.cm.gray,vmin=0,vmax=255)
plt.subplot(4,5,20)
plt.imshow(np.absolute(img4),cmap=plt.cm.gray,vmin=0,vmax=255)
plt.show()
