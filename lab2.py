#importowanie bibliotek
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc as mc


#funkcja liczaca histogram
def hist(img,N):
    H=np.zeros(N)
    #print H
    p=np.linspace(0,255,N+1)
    for i in range (0,N-1):
        H[i]=np.sum((img>=p[i])&(img<p[i+1]))
    H[N-1]=np.sum((img>=p[N-1])&(img<=p[N]))
    return H
    
    
#
#funkcja do zmiany obrazu
def zm(y,LUT,x1,x2):
    y1=y.copy()
    m ,n=y.shape
    #zmiana wartosci w obrazie
    for i in range (x1+1,x2+1):
        for j in range (0,m):
            for k in range(0,n):
                if y[j][k]==i-1:
                    y1[j][k]=LUT[i]
    return y1
#
#importowanie obrazu
img = mc.imread('kierowca.png')
r=img[:,:,0]
g=img[:,:,1]
b=img[:,:,2]
y=0.2126*r+0.7152*g+0.0722*b
#rozmiary obrazu
m, n=y.shape
N=256
plt.subplot(3,7,1)
plt.imshow(y,cmap=plt.cm.gray,vmin=0, vmax=255)
#histogram oryginalu
H=hist(y,N)
#normalizacja do wyswietlenia
Hp=H.copy()/np.sum(H)
plt.subplot(3,7,8)
plt.bar(range(len(Hp)),Hp,0.5,color="black")
#wypelnienie LUT - pierwsze
LUT=np.linspace(0,255,256)
plt.subplot(3,7,15)
plt.plot(LUT)
#max wartosc w obrazie
Hp=np.where(H>0)
x1=np.min(Hp)
x2=np.max(Hp)
#linowe
#wypelnienie LUT
LUT=np.zeros(N)
for i in range (x1,x2+1):
    LUT[i]=(255/(x2-x1))*(i-x1)
for i in range (x2+1,N):
    LUT[i]=255
print LUT
plt.subplot(3,7,16)
plt.plot(LUT)
y1=y.copy()
#zmiana wartosci w obrazie
for i in range (x1+1,x2+1):
        y1[y==i-1]=LUT[i]
plt.subplot(3,7,2)
plt.imshow(y1, cmap=plt.cm.gray, vmin=0, vmax=255)
#liczenie histogramu
H1=hist(y1,N)
Hp=H1.copy()/np.sum(H1)
plt.subplot(3,7,9)
plt.bar(range(len(Hp)),Hp,0.5,color="black")
#potegowanie
LUT=np.zeros(N)
#stworzenia wektora  x od 0 do sqrt(255), x jest x2-x1+1
x=np.linspace(0,np.power(255,0.5),x2-x1+1)
LUT[x1:x2+1]=np.round(np.power(x,2))
for i in range (x2+1,N):
    LUT[i]=255
print LUT
plt.subplot(3,7,17)
plt.plot(LUT)
y1=y.copy()
#zmiana wartosci w obrazie
for i in range (x1+1,x2+1):
        y1[y==i-1]=LUT[i]
plt.subplot(3,7,3)
plt.imshow(y1, cmap=plt.cm.gray, vmin=0, vmax=255)
#liczenie histogramu
H1=hist(y1,N)
Hp=H1.copy()/np.sum(H1)
plt.subplot(3,7,10)
plt.bar(range(len(Hp)),Hp,0.5,color="black")
#pierwiastkowanie
LUT=np.zeros(N)
#stworzenie wektora od 0 do 255^2,x jest x2-x1+1
x=np.linspace(0,np.power(255,2),x2-x1+1)
LUT[x1:x2+1]=np.round(np.power(x,0.5))
for i in range (x2+1,N):
    LUT[i]=255
print LUT
plt.subplot(3,7,18)
plt.plot(LUT)
y2=y.copy()
#zmiana wartosci w obrazie
for i in range (x1+1,x2+1):
        y2[y==i-1]=LUT[i]
plt.subplot(3,7,4)
plt.imshow(y2, cmap=plt.cm.gray, vmin=0, vmax=255)
#liczenie histogramu
H1=hist(y2,N)
Hp=H1.copy()/np.sum(H1)
plt.subplot(3,7,11)
plt.bar(range(len(Hp)),Hp,0.5,color="black")
#logarytmowanie
LUT=np.zeros(N)
#stworzenie wektora od 0 do 255
tmp = np.linspace(0,255,x2-x1+1)
#logarytm nat z tmp
tmp2 = np.log(tmp[1:tmp.size])
LUT[x1+1:x2+1] = np.round(tmp2/np.max(tmp2)*255)
for i in range (x2+1,N):
    LUT[i]=255
print LUT
plt.subplot(3,7,19)
plt.plot(LUT)
y1=y.copy()
#zmiana wartosci w obrazie
for i in range (x1+1,x2+1):
        y1[y==i-1]=LUT[i]
plt.subplot(3,7,5)
plt.imshow(y1, cmap=plt.cm.gray, vmin=0, vmax=255)
#liczenie histogramu
H1=hist(y1,N)
Hp=H1.copy()/np.sum(H1)
plt.subplot(3,7,12)
plt.bar(range(len(Hp)),Hp,0.5,color="black")
#2^x
LUT=np.zeros(N)
LUT[x1:x2+1] = np.round(np.exp(np.linspace(0, np.log(255), x2-x1+1)))
for i in range (x2+1,N):
    LUT[i]=255
print LUT
plt.subplot(3,7,20)
plt.plot(LUT)
y1=y.copy()
#zmiana wartosci w obrazie
for i in range (x1+1,x2+1):
        y1[y==i-1]=LUT[i]
plt.subplot(3,7,6)
plt.imshow(y1, cmap=plt.cm.gray, vmin=0, vmax=255)
#liczenie histogramu
H1=hist(y1,N)
Hp=H1.copy()/np.sum(H1)
plt.subplot(3,7,13)
plt.bar(range(len(Hp)),Hp,0.5,color="black")
#wyrownanie histogramu
d=np.zeros(N)
for i in range(0,N):
    for j in range (0,i):
        #dystrybuanta
        d[i] =d[i]+H[j]
dmin = np.where(d > 0)
dmin = np.min(dmin)
hv = np.zeros(N)
for i in range(0,N):
        hv[i] = np.round(((d[i] - dmin) / (m * n - dmin)) * 255)
print hv
plt.subplot(3,7,21)
plt.plot(hv)
y1=y.copy()
#zmiana wartosci w obrazie
for i in range (x1+1,x2+1):
        y1[y==i-1]=hv[i]
plt.subplot(3,7,7)
plt.imshow(y1, cmap=plt.cm.gray, vmin=0, vmax=255)
#liczenie histogramu
H1=hist(y1,N)
Hp=H1.copy()/np.sum(H1)
plt.subplot(3,7,14)
plt.bar(range(len(Hp)),Hp,0.5,color="black")
plt.show()
