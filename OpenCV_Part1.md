# OpenCV
## Part 1
```python
import numpy as np
import matplotlib.pyplot as plt
import cv2
```


```python
# 1. Load image
img = cv2.imread("Mushroom.jpg")
type(img)
```




    numpy.ndarray




```python
img_wrong = cv2.imread('wrong/path/doesnot/abcdegh.jpg')

type(img_wrong)
```




    NoneType




```python
plt.imshow(img)
```




    <matplotlib.image.AxesImage at 0x20bb3ef5550>




    
![png](output_3_1.png)
    



```python
# 2. Display original (convert BGR → RGB)
fix_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(fix_img)
```




    <matplotlib.image.AxesImage at 0x20bb6feee90>




    
![png](output_4_1.png)
    



```python
# 3. Load grayscale version
img_gray = cv2.imread("Mushroom.jpg", cv2.IMREAD_GRAYSCALE)
img_gray.shape
plt.imshow(img_gray)
plt.imshow(img_gray, cmap="gray")
```




    <matplotlib.image.AxesImage at 0x20bb7079e50>




    
![png](output_5_1.png)
    



```python
# 4. Resize to specific dimensions
fix_img.shape
new_img = cv2.resize(fix_img, (1000, 400))
plt.imshow(new_img)
```




    <matplotlib.image.AxesImage at 0x20bb6847c50>




    
![png](output_6_1.png)
    



```python
# 5. Resize by scale factors
new_img.shape
w_ratio = 0.5
h_ratio = 0.5
new_img = cv2.resize(fix_img, (0, 0), fx=w_ratio, fy=h_ratio)
plt.imshow(new_img)
```




    <matplotlib.image.AxesImage at 0x20bb68e4410>




    
![png](output_7_1.png)
    



```python
# 6. Flip vertically
flip_img = cv2.flip(fix_img, 0)
plt.imshow(flip_img)
```




    <matplotlib.image.AxesImage at 0x20bb7067110>




    
![png](output_8_1.png)
    



```python
# 7. Flip both vertically and horizontally
flip_img2 = cv2.flip(fix_img, -1)
plt.imshow(flip_img2)
```




    <matplotlib.image.AxesImage at 0x20bbbe30f50>




    
![png](output_9_1.png)
    



```python
# 8. Save flipped image
type(fix_img)
cv2.imwrite('Mushroom_fixed_image.jpg', flip_img)
```




    True




```python

```
