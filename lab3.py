#importowanie bibliotek
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc as mc
#funkcja liczaca macierz wspolwystepowania
def fCoocurence(img,N,dir1):
    macierz=np.zeros((N,N))
    w1, k1=img.shape
    '''
    img=img[0:w1+dir1[0]:1,0:k1+dir1[1]:1]
    for i in range (0,255):
        W, K=np.where(img==i)
        #print W.shape
        [m]=W.shape
        [l]=K.shape
        for w in range (0,m):
            for k in range (0,l):
                if ((w+dir1[0] < w1) and (k+dir1[1] < k1) and (w+dir1[0] >0) and (k+dir1[1] >0)):
                    x = img[w,k]
                    y = img[w+dir1[0],k+dir1[1]]
                    x=x.astype('uint8')
                    y=y.astype('uint8')
                    macierz[x,y] = macierz[x,y]+1
                    '''
    W=np.zeros((w1,k1))
    p=np.linspace(0,255,N+1)
    for i in range(0,N):
        a,b=np.where((img>p[i-1])&(img<p[i+1]))
        W[a,b]=i
    #wybor kierunku - dziala tylko dla 0,1 - lewo 
    if (dir1==[0,1]):
        W1=W[0:w1,0:k1-1]
        W2=W[0:w1,1:k1]
    if (dir1==[0,-1]):
        W2=W[w1:0:-1,k1-1:0:-1]
        #H2=np.fliplr(H2)
        W1=W[w1:0:-1,k1:1:-1]
        #H1=np.fliplr(H1)
        print W1
        print W2
    if (dir1==[1,0]):
        W1=W[0:w1-1,0:k1]
        W2=W[1:w1,0:k1]
        print W1
        print W2
        W1=np.rot90(W1)
        W1=np.rot90(W1)
        W2=np.rot90(W2)
        W2=np.rot90(W2)
        print W1
        print W2
    if (dir1==[-1,0]):
        W1=W[w1-1:0:-1,k1:0:-1]
        W2=W[w1:1:-1,k1:0:-1]
        print W1
        print W2
        W1=np.rot90(W1)
        W1=np.rot90(W1)
        print W1
        print W2
        W2=np.rot90(W2)
        W2=np.rot90(W2)
    #
    #
    r, t=W2.shape
    for i in range(0,r):
        for j in range(0,t):
           x=W1[i,j]
           x=x.astype('uint8')
           y=W2[i,j]
           y=y.astype('uint8')
           macierz[x,y]=macierz[x,y]+1;
    return macierz


#
#funkcja do oceny rozmycia obrazu
def rozmycie1(img):
    w,k=img.shape
    a=0
    for i in range (0,w):
        a=a+img[i,i]
    return a


#
def rozmycie2(img):
    w,k=img.shape
    a=np.sum(img>0)
    return a


#
#importowanie obrazu
img1 = mc.imread('pic1.png')
img2=mc.imread('pic2.png')
#print img1.shape
#zmiana na szarosci
r1=img1[:,:,0]
g1=img1[:,:,1]
b1=img1[:,:,2]
y1=0.2126*r1+0.7152*g1+0.0722*b1
plt.subplot(2,2,1)
plt.imshow(y1,cmap=plt.cm.gray,vmin=0, vmax=255)
r2=img2[:,:,0]
g2=img2[:,:,1]
b2=img2[:,:,2]
y2=0.2126*r2+0.7152*g2+0.0722*b2
plt.subplot(2,2,3)
plt.imshow(y2,cmap=plt.cm.gray,vmin=0, vmax=255)
#liczenie macierzy wspolwystepowania
#kierunek relacji
xy1=[0,1]
xy2=[0,-1]
xy3=[1,0]
xy4=[-1,0]
#0,1
mw1=fCoocurence(y1,256,xy1)
mw2=fCoocurence(y2,256,xy1)
plt.subplot(2,2,2)
#print mw1
plt.imshow(mw1,cmap=plt.cm.gray,vmin=0, vmax=255)
plt.subplot(2,2,4)
#print mw2
plt.imshow(mw2,cmap=plt.cm.gray,vmin=0, vmax=255)
print "Stopien rozmycia jako suma wartosci pikseli na przekatnej. \nIm wiecej tym bardziej rozmyte : "
print rozmycie1(mw1)
print rozmycie1(mw2)
print "Stopien rozmycia jako ilosc pikseli roznych od 0. \nIm mniej tym bardziej rozmyte : "
print rozmycie2(mw1)
print rozmycie2(mw2)
'''
#0,-1
mw1=fCoocurence(y1,10,xy2)
mw2=fCoocurence(y2,10,xy2)
plt.subplot(2,5,3)
#print mw1
plt.imshow(mw1,cmap=plt.cm.gray,vmin=0, vmax=255)
plt.subplot(2,5,8)
#print mw2
plt.imshow(mw2,cmap=plt.cm.gray,vmin=0, vmax=255)
print rozmycie(mw1)
print rozmycie(mw2)
#1,0
mw1=fCoocurence(y1,10,xy3)
mw2=fCoocurence(y2,10,xy3)
plt.subplot(2,5,4)
#print mw1
plt.imshow(mw1,cmap=plt.cm.gray,vmin=0, vmax=255)
plt.subplot(2,5,9)
#print mw2
plt.imshow(mw2,cmap=plt.cm.gray,vmin=0, vmax=255)
print rozmycie(mw1)
print rozmycie(mw2)
#-1,0
mw1=fCoocurence(y1,10,xy4)
mw2=fCoocurence(y2,10,xy4)
plt.subplot(2,5,5)
#print mw1
plt.imshow(mw1,cmap=plt.cm.gray,vmin=0, vmax=255)
plt.subplot(2,5,10)
#print mw2
plt.imshow(mw2,cmap=plt.cm.gray,vmin=0, vmax=255)
#rozmycie
print rozmycie(mw1)
print rozmycie(mw2)
'''
plt.show()
