```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
```


```python
flat_chess = cv2.imread('chessboard_green.png')
flat_chess = cv2.cvtColor(flat_chess,cv2.COLOR_BGR2RGB)
plt.imshow(flat_chess)
```




    <matplotlib.image.AxesImage at 0x19600fd9160>




    
![png](output_1_1.png)
    



```python
#Load this in gray scale

gray_flat_chess = cv2.cvtColor(flat_chess,cv2.COLOR_BGR2GRAY)
plt.imshow(gray_flat_chess,cmap='gray')
```




    <matplotlib.image.AxesImage at 0x19601095e50>




    
![png](output_2_1.png)
    



```python
# Uploading real chessboard image

real_chess = cv2.imread('Chessboard.jpg')
real_chess = cv2.cvtColor(real_chess,cv2.COLOR_BGR2RGB)
plt.imshow(real_chess)
```




    <matplotlib.image.AxesImage at 0x196018b0910>




    
![png](output_3_1.png)
    



```python
gray_real_chess = cv2.cvtColor(real_chess,cv2.COLOR_BGR2GRAY)
plt.imshow(gray_real_chess,cmap='gray')
```




    <matplotlib.image.AxesImage at 0x19601913c50>




    
![png](output_4_1.png)
    



```python
#Harris Corner Detection

gray = np.float32(gray_flat_chess)
dst = cv2.cornerHarris( src=gray,blockSize=2,ksize=3,k=0.04)

dst = cv2.dilate(dst,None)
```


```python
flat_chess[dst>0.01*dst.max()] = [255,0,0]
plt.imshow(flat_chess)

```




    <matplotlib.image.AxesImage at 0x196019a2990>




    
![png](output_6_1.png)
    



```python
gray = np.float32(gray_real_chess)
dst = cv2.cornerHarris(src=gray,blockSize=2,ksize=3,k=0.04)
dst = cv2.dilate(dst,None)

real_chess[dst>0.01*dst.max()] = [255,0,0]
plt.imshow(real_chess) 
```




    <matplotlib.image.AxesImage at 0x19601a387d0>




    
![png](output_7_1.png)
    



```python
#Shi-Tomasi Corner Detection

corners = cv2.goodFeaturesToTrack(gray_flat_chess,64,0.01,10)

corners = np.int32(corners)
```


```python
for i in corners:
  x,y = i.ravel()
  cv2.circle(flat_chess,(x,y),3,(255,0,0),-1)

plt.imshow(flat_chess)
```




    <matplotlib.image.AxesImage at 0x19603e08190>




    
![png](output_9_1.png)
    



```python
corners = cv2.goodFeaturesToTrack(gray_real_chess,100,0.01,10)

corners = np.int32(corners)

for i in corners:
  x,y = i.ravel()
  cv2.circle(real_chess,(x,y),3,
             (255,0,0),-1)
plt.imshow(real_chess)
```




    <matplotlib.image.AxesImage at 0x19603e6df90>




    
![png](output_10_1.png)
    



```python

```
