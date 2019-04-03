#importowanie bibliotek
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc as mc
#funkcja wyliczajaca jasnosc i kontrast
def jasnosc(img):
	M, N = img.shape
	j=np.sum(img)/(M*N)
	k=np.sum(np.power(img-j,2))
	k=k/(M*N)
	k=np.power(k,0.5)
	return j,k


#
#funkcja wyliczajaca jasnosc - moje
def jas(img):
        M, N=img.shape
        
        return j


# 
#funkcja wyliczajaca kontrast - moje
def kont(img,j):
        M, N=img.shape
        k=np.sum(img-j)
        k=k/(M*N)
        return k


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
#importowanie obrazu
img = mc.imread('kierowca.png')
#kopia obrazu w formacie float
img2 = img.copy()
img2 = img2.astype('float64')
#
r=img[:,:,0]
g=img[:,:,1]
b=img[:,:,2]
y=0.2126*r+0.7152*g+0.0722*b
plt.subplot(2,6,1)
plt.imshow(y,cmap=plt.cm.gray,vmin=0, vmax=255)
y=y.astype('uint8')
#
r2=img2[:,:,0]
g2=img2[:,:,1]
b2=img2[:,:,2]
y0=0.2126*r2+0.7152*g2+0.0722*b2
plt.subplot(2,6,7)
plt.imshow(y0,cmap=plt.cm.gray,vmin=0, vmax=255)
y0=y0.astype('float64')
#kontrast i jasnosc oryginalnego obrazu
print ('\nJasnosc i kontrast obrazu oryginalnego:')
print y.dtype
print jasnosc (y)
print y0.dtype
print jasnosc (y0)
#dodanie stalej do obrazu
print ('\nJasnosc i kontrast obrazu po dodaniu stalej:')
y1=y.copy()
print y1.dtype
y1=y1+50.5
y1=czy(y1)
plt.subplot(2,6,2)
plt.imshow(y1,cmap=plt.cm.gray,vmin=0, vmax=255)
print y1.dtype
print jasnosc (y1)
y11=y0.copy()
print y11.dtype
y11=y11+50.5
y11=czy(y11)
plt.subplot(2,6,8)
plt.imshow(y11,cmap=plt.cm.gray,vmin=0, vmax=255)
print y11.dtype
print jasnosc (y11)
#mnozenie przez stala
print ('\nJasnosc i kontrast obrazu po mnozeniu przez stala:')
y2=y.copy()
print y2.dtype
y2=y2*50.5
y2=czy(y2)
plt.subplot(2,6,3)
plt.imshow(y2,cmap=plt.cm.gray,vmin=0, vmax=255)
print y2.dtype
print jasnosc(y2)
y12=y0.copy()
print y12.dtype
y12=y12*50.5
y12=czy(y12)
plt.subplot(2,6,9)
plt.imshow(y12,cmap=plt.cm.gray,vmin=0, vmax=255)
print y12.dtype
print jasnosc(y12)
#potegowanie obrazu
print ('\nJasnosc i kontrast obrazu po potegowaniu:')
y3=y.copy()
print y3.dtype
y3=np.power(y3,4.5)
y3=czy(y3)
plt.subplot(2,6,4)
plt.imshow(y3,cmap=plt.cm.gray,vmin=0, vmax=255)
print y3.dtype
print jasnosc(y3)
y13=y0.copy()
print y13.dtype
y13=np.power(y13,4.5)
y13=czy(y13)
plt.subplot(2,6,10)
plt.imshow(y13,cmap=plt.cm.gray,vmin=0, vmax=255)
print y13.dtype
print jasnosc(y13)
#pierwiastkowanie
print('\nJasnosc i kontrast obrazu po pierwiastkowaniu:')
y4=y.copy()
print y4.dtype
y4=np.power(y4,0.8)
y4=czy(y4)
plt.subplot(2,6,5)
plt.imshow(y4,cmap=plt.cm.gray,vmin=0,vmax=255)
print y4.dtype
print jasnosc(y4)
y14=y0.copy()
print y14.dtype
y14=np.power(y14,0.8)
y14=czy(y14)
plt.subplot(2,6,11)
plt.imshow(y14,cmap=plt.cm.gray,vmin=0,vmax=255)
print y14.dtype
print jasnosc(y14)
#logarytmowanie obrazu
print ('\nJasnosc i kontrast obrazu po logarytmowaniu:')
y5=y.copy()
print y5.dtype
x1=5
x1=np.log(x1)
x1=x1.astype('uint8')
y5=np.log(y5)
y5=y5.astype('uint8')
y5=y5/x1
y5=czy(y5)
plt.subplot(2,6,6)
plt.imshow(y5,cmap=plt.cm.gray,vmin=0, vmax=255)
print y5.dtype
print jasnosc(y5)
y15=y0.copy()
print y15.dtype
x2=5.5
x2=np.log(x2)
y15=np.log(y15)
y15=y15/x2
y15=czy(y15)
plt.subplot(2,6,12)
plt.imshow(y15,cmap=plt.cm.gray,vmin=0, vmax=255)
print y15.dtype
print jasnosc(y15)
plt.show()
