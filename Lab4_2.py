import numpy as np
import matplotlib.pyplot as plt
from scipy import misc as mc

def     ffilter(img,mask):
        (m,n)=mask.shape
        (a,b)=img.shape
        w1=m/2
        w2=m-(w1+1)
        w3=n/2
        w4=n-(w3+1)
        A=img[w1:0:-1,0:b]
        B=img[a:a-w2-1:-1,0:b]
        img2=np.concatenate((A,img,B), axis=0)
        C=img2[0:a+w1+w2,w3:0:-1]
        D=img2[0:a+w1+w2,b:b-w4-1:-1]
        img_new=np.concatenate((C,img2,D), axis=1)
        N=np.sum(mask)
        if np.sum(mask)==0:
            N=1
        s=np.zeros((a,b))
        for i in range(0,a):
            for j in range(0,b):
                s[i][j]=(np.sum(img_new[i:i+m,j:j+n]*mask))/N
        return s


img = mc.imread('litery_1.png')
w=img[:,:,0]

#Rozszerzenie obrazu (odbicie lustrzane)

plt.subplot(2,3,1)
plt.imshow(w, cmap=plt.cm.gray, vmin=0, vmax=255)

#Filtr Robertsa

rmask1=np.array([[-1,0],[1,0]])
wr1=ffilter(w.copy(),rmask1)

rmask2=np.array([[-1,1],[0,0]])
wr2=ffilter(w.copy(),rmask2)

rmask3=np.array([[0,1],[-1,0]])
wr3=ffilter(w.copy(),rmask3)

rmask4=np.array([[1,0],[0,-1]])
wr4=ffilter(w.copy(),rmask4)

wr=np.sqrt(np.power(wr1,2)+np.power(wr2,2)+np.power(wr3,2)+np.power(wr4,2))

plt.subplot(2,3,2)
plt.imshow(wr, cmap=plt.cm.gray, vmin=0, vmax=255)

#Filtr Laplace'a

lmask1=np.array([[0,1,0],[1,-4,1],[0,1,0]])
wl1=ffilter(w.copy(),lmask1)

lmask2=np.array([[1,1,1],[1,-8,1],[1,1,1]])
wl2=ffilter(w.copy(),lmask2)

wl=np.sqrt(np.power(wl1,2)+np.power(wl2,2))

plt.subplot(2,3,3)
plt.imshow(wl, cmap=plt.cm.gray, vmin=0, vmax=255)

#Filtr Prewitta

pmask1=np.array([[0,1,0],[1,-4,1],[0,1,0]])
wp1=ffilter(w.copy(),pmask1)

pmask2=np.array([[1,1,1],[1,-8,1],[1,1,1]])
wp2=ffilter(w.copy(),pmask2)

pmask3=np.array([[0,1,0],[1,-4,1],[0,1,0]])
wp3=ffilter(w.copy(),pmask3)

pmask4=np.array([[1,1,1],[1,-8,1],[1,1,1]])
wp4=ffilter(w.copy(),pmask4)

wp=np.sqrt(np.power(wp1,2)+np.power(wp2,2)+np.power(wp3,2)+np.power(wp4,2))

plt.subplot(2,3,4)
plt.imshow(wp, cmap=plt.cm.gray, vmin=0, vmax=255)

#Filtr Sobela

smask1=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
ws1=ffilter(w.copy(),smask1)

smask2=np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
ws2=ffilter(w.copy(),smask2)

smask3=np.array([[0,1,2],[-1,0,1],[-2,-1,0]])
ws3=ffilter(w.copy(),smask3)

smask4=np.array([[-2,-1,0],[-1,0,1],[0,1,2]])
ws4=ffilter(w.copy(),smask4)

ws=np.sqrt(np.power(ws1,2)+np.power(ws2,2)+np.power(ws3,2)+np.power(ws4,2))

plt.subplot(2,3,5)
plt.imshow(ws, cmap=plt.cm.gray, vmin=0, vmax=255)

#Filtr Kirscha

kmask1=np.array([[-3,-3,5],[-3,0,5],[-3,-3,5]])
wk1=ffilter(w.copy(),kmask1)

kmask2=np.array([[-3,5,5],[-3,0,5],[-3,-3,-3]])
wk2=ffilter(w.copy(),kmask2)

kmask3=np.array([[5,5,5],[-3,0,-3],[-3,-3,-3]])
wk3=ffilter(w.copy(),kmask3)

kmask4=np.array([[5,5,-3],[5,0,-3],[-3,-3,-3]])
wk4=ffilter(w.copy(),kmask4)

kmask5=np.array([[5,-3,-3],[5,0,-3],[5,-3,-3]])
wk5=ffilter(w.copy(),kmask5)

kmask6=np.array([[-3,-3,-3],[5,0,-3],[5,5,-3]])
wk6=ffilter(w.copy(),kmask6)

kmask7=np.array([[-3,-3,-3],[-3,0,-3],[5,5,5]])
wk7=ffilter(w.copy(),kmask7)

kmask8=np.array([[-3,-3,-3],[-3,0,5],[-3,5,5]])
wk8=ffilter(w.copy(),kmask8)

wk=np.sqrt(np.power(wk1,2)+np.power(wk2,2)+np.power(wk3,2)+np.power(wk4,2)+np.power(wk5,2)+np.power(wk6,2)+np.power(wk7,2)+np.power(wk8,2))

plt.subplot(2,3,6)
plt.imshow(wk, cmap=plt.cm.gray, vmin=0, vmax=255)

plt.show()

