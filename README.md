## Computer_Vision_Part1
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

## Computer_Vision_Part2
```python
import cv2
import matplotlib.pyplot as plt
```


```python
img = cv2.imread("Mushroom.jpg")

plt.imshow(img)
```




    <matplotlib.image.AxesImage at 0x1ad0bc99160>




    
![png](output_1_1.png)
    



```python
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img1)
```




    <matplotlib.image.AxesImage at 0x1ad0cd2a0d0>




    
![png](output_2_1.png)
    



```python
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

plt.imshow(img2)

```




    <matplotlib.image.AxesImage at 0x1ad0cdbca50>




    
![png](output_3_1.png)
    



```python
img3 = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

plt.imshow(img3)
```




    <matplotlib.image.AxesImage at 0x1ad0ce32ad0>




    
![png](output_4_1.png)
    



```python
img1 = cv2.imread('do-not-copy-stamp.jpg')
img2 = cv2.imread("Mushroom.jpg")

plt.imshow(img1)
```




    <matplotlib.image.AxesImage at 0x1ad17a0c910>




    
![png](output_5_1.png)
    



```python
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
```


```python
plt.imshow(img1)
```




    <matplotlib.image.AxesImage at 0x1ad110b0b90>




    
![png](output_7_1.png)
    



```python
plt.imshow(img2)
```




    <matplotlib.image.AxesImage at 0x1ad11158050>




    
![png](output_8_1.png)
    



```python
img1 = cv2.resize(img1, (1200,1200))
img2 = cv2.resize(img2, (1200, 1200))

alpha = 0.5
beta = 0.5

blended = cv2.addWeighted(img1, alpha, img2, beta, gamma=0)

plt.imshow(blended)
```




    <matplotlib.image.AxesImage at 0x1ad1373de50>




    
![png](output_9_1.png)
    



```python
alpha = 0.8
beta = 0.2

blended1 = cv2.addWeighted(img1, alpha, img2, beta, 0)
plt.imshow(blended1)
```




    <matplotlib.image.AxesImage at 0x1ad1378fc50>




    
![png](output_10_1.png)
    



```python
alpha = 0.2
beta = 0.8

blended1 = cv2.addWeighted(img1, alpha, img2, beta, 0)
plt.imshow(blended1)
```




    <matplotlib.image.AxesImage at 0x1ad13825a90>




    
![png](output_11_1.png)
    



```python
img1 = cv2.imread('do-not-copy-stamp.jpg')
img2 = cv2.imread('Mushroom.jpg')
```


```python
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
```


```python
img1 = cv2.resize(img1, (200,200))

large_img = img2
small_img = img1

x_offset = 0
y_offset = 0

x_end = x_offset + small_img.shape[1]
y_end = y_offset + small_img.shape[0]

large_img[y_offset:y_end, x_offset:x_end] = small_img

plt.imshow(large_img)
```




    <matplotlib.image.AxesImage at 0x1ad111e7890>




    
![png](output_14_1.png)
    



```python

```
## Computer_Vision_Part3
```python
import cv2
import matplotlib.pyplot as plt
```


```python
img = cv2.imread('rainbow.jpg')
plt.imshow(img)
```




    <matplotlib.image.AxesImage at 0x1eea6611160>




    
![png](output_1_1.png)
    



```python
img = cv2.imread('rainbow.jpg', 0)
plt.imshow(img, cmap = 'gray')
```




    <matplotlib.image.AxesImage at 0x1eea80d25d0>




    
![png](output_2_1.png)
    



```python
ret1, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

ret1
thresh1

plt.imshow(thresh1, cmap = 'gray')
```




    <matplotlib.image.AxesImage at 0x1eea8144a50>




    
![png](output_3_1.png)
    



```python
img2 = cv2.imread('rainbow.jpg', 0)
ret1, thresh1 = cv2.threshold(img2, 127, 255, cv2.THRESH_TRUNC)
plt.imshow(thresh1, cmap = "gray")
```




    <matplotlib.image.AxesImage at 0x1eea92be350>




    
![png](output_4_1.png)
    



```python
img3 = cv2.imread('rainbow.jpg', 0)
ret1, thresh1 = cv2.threshold(img3, 127, 255, cv2.THRESH_TOZERO)
plt.imshow(thresh1, cmap = "gray")
```




    <matplotlib.image.AxesImage at 0x1eea930f9d0>




    
![png](output_5_1.png)
    



```python
img_r = cv2.imread('crossword.jpg', 0)
plt.imshow(img_r, cmap = 'gray')
```




    <matplotlib.image.AxesImage at 0x1eeae4e4190>




    
![png](output_6_1.png)
    



```python
def show_pic(img):
    fig = plt.figure(figsize = (15,15))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap = 'gray')

show_pic(img_r)
```


    
![png](output_7_0.png)
    



```python
ret, th1 = cv2.threshold(img_r, 127, 255, cv2.THRESH_BINARY)
show_pic(th1)
```


    
![png](output_8_0.png)
    



```python
ret, th1 = cv2.threshold(img_r, 200, 255, cv2.THRESH_BINARY)
show_pic(th1)
```


    
![png](output_9_0.png)
    



```python
th2 = cv2.adaptiveThreshold(img_r, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 8)
show_pic(th2)
```


    
![png](output_10_0.png)
    



```python
blended = cv2.addWeighted(src1 = th1, alpha = 0.6,
                          src2 = th2, beta = 0.4, gamma = 0)

show_pic(blended)
```


    
![png](output_11_0.png)
    



```python
th3 = cv2.adaptiveThreshold(img_r, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 8)

blended = cv2.addWeighted(src1=th1, alpha=0.6,
                          src2=th2, beta=0.4, gamma=0)

show_pic(blended)
```


    
![png](output_12_0.png)
    



```python

```
# Aspect Detection
## Corner Detection
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
## Edge Detection
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
```


```python
img = cv2.imread('Mushroom.jpg')
plt.imshow(img) #here we are not concerned about image color change as we are performing edge detection
```




    <matplotlib.image.AxesImage at 0x216a6d6efd0>




    
![png](output_1_1.png)
    



```python
edges = cv2.Canny(image=img,
                  threshold1=127,
                  threshold2=127) #setting both threshold as median values
plt.imshow(edges)
```




    <matplotlib.image.AxesImage at 0x216a6de4e10>




    
![png](output_2_1.png)
    



```python
med_value = np.median(img)
med_value

#lower and upper limit of threshold

lower = int(max(0,0.7*med_value)) #lower threshold = 0 or 70% of median value whichever is greater
upper = int(min(255,1.3*med_value)) #upper threshold is 30% above threshold or 255 whichever is smaller
```


```python
edges = cv2.Canny(image=img,
                  threshold1=lower,
                  threshold2=upper)
plt.imshow(edges)
```




    <matplotlib.image.AxesImage at 0x216a751ae90>




    
![png](output_4_1.png)
    



```python
edges = cv2.Canny(image=img,
                  threshold1=lower,
                  threshold2=upper+100)
plt.imshow(edges)
```




    <matplotlib.image.AxesImage at 0x216a7598cd0>




    
![png](output_5_1.png)
    



```python
blurred_img = cv2.blur(img,ksize=(5,5))

edges = cv2.Canny(image=blurred_img,
                  threshold1=lower,
                  threshold2=upper)
plt.imshow(edges)
```




    <matplotlib.image.AxesImage at 0x216a2f72c10>




    
![png](output_6_1.png)
    



```python

```
# Feature Detection
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
```


```python
def display(img, cmap='gray'):
    # Display an image using Matplotlib
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap=cmap)
    plt.show()
```


```python
# Load images in grayscale
reeses = cv2.imread('reeses_puffs.png', 0)
display(reeses)
```


    
![png](output_2_0.png)
    



```python
cereals = cv2.imread('many_cereals.jpg', 0)
display(cereals)
```


    
![png](output_3_0.png)
    



```python
# Create ORB detector
orb = cv2.ORB_create()
```


```python
# Detect keypoints and descriptors using ORB
kp1, des1 = orb.detectAndCompute(reeses, mask=None)
kp2, des2 = orb.detectAndCompute(cereals, mask=None)
```


```python
# Match descriptors using Brute Force Matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)  # Match descriptors
```


```python
# Sort matches based on distance
matches = sorted(matches, key=lambda x: x.distance)
```


```python
# Draw top 25 matches
reeses_matches = cv2.drawMatches(reeses, kp1, cereals, kp2, matches[:25], None, flags=2)
display(reeses_matches)
```


    
![png](output_8_0.png)
    



```python
# Create SIFT detector (corrected)
sift = cv2.SIFT_create()
```


```python
# Detect keypoints and descriptors using SIFT
kp1, des1 = sift.detectAndCompute(reeses, None)
kp2, des2 = sift.detectAndCompute(cereals, None)
```


```python
# Create SIFT detector (corrected)
sift = cv2.SIFT_create()
```


```python
# Detect keypoints and descriptors using SIFT
kp1, des1 = sift.detectAndCompute(reeses, None)
kp2, des2 = sift.detectAndCompute(cereals, None)
```


```python
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
```


```python
good =[]
# LESS DISTANCE -- BETTER THE MATCH
for match1,match2 in matches:
  #if MATCH1 DISTANCE is < 75% of MATCH 2 THEN IT IS A GOOD MATCH
  if match1.distance < 0.75*match2.distance:
    good.append([match1])
```


```python
print('Length of total matches: ',len(matches))
```

    Length of total matches:  1501
    


```python
print('Length of good matches: ',len(good))
```

    Length of good matches:  79
    


```python
sift_matches = cv2.drawMatchesKnn(reeses,kp1,cereals,kp2,good,None,flags=2)
display(sift_matches)
```


    
![png](output_17_0.png)
    



```python
# Initialize the SIFT detector
sift = cv2.SIFT_create()
```


```python
# Detect keypoints and descriptors in both images
kp1, des1 = sift.detectAndCompute(reeses, None)
kp2, des2 = sift.detectAndCompute(cereals, None)
```


```python
flann_index_KDtree = 0
index_params = dict(algorithm=flann_index_KDtree,trees=5)
search_params = dict(checks=50)
```


```python
flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1,des2,k=2)
```


```python
good =[]
# LESS DISTANCE -- BETTER THE MATCH
for match1,match2 in matches:
  #if MATCH1 DISTANCE is < 75% of MATCH 2 THEN IT IS A GOOD MATCH
  if match1.distance < 0.75*match2.distance:
    good.append([match1])
```


```python
flann_matches = cv2.drawMatchesKnn(reeses,kp1,cereals,kp2,good,None,flags=0)
display(flann_matches)
```


    
![png](output_23_0.png)
    



```python
#with mask

sift = cv2.SIFT_create()

kp1,des1 = sift.detectAndCompute(reeses,None)
kp2,des2 = sift.detectAndCompute(cereals,None)

flann_index_KDtree = 0
index_params = dict(algorithm=flann_index_KDtree,trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1,des2,k=2)
```


```python
#masking
matchesMask = [[0,0] for i in range(len(matches))] #bunch of zeros pure black and some will be on

# LESS DISTANCE -- BETTER THE MATCH
for i,(match1,match2) in enumerate(matches):
  #if MATCH1 DISTANCE is < 75% of MATCH 2 THEN IT IS A GOOD MATCH
  if match1.distance < 0.75*match2.distance:
    matchesMask[i] = [1,0]

draw_params = dict(matchColor=(0,255,0),
                   singlePointColor=(255,0,0),
                   matchesMask=matchesMask,
                   flags=0) #by changing flags = 2 we can remove the red dots

flann_matches = cv2.drawMatchesKnn(reeses,kp1,cereals,kp2,matches,None,**draw_params)
display(flann_matches)
```


    
![png](output_25_0.png)
    



```python

```
# Object Detection
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
```


```python
full = cv2.imread('Training_Sunflower.jpg')
full = cv2.cvtColor(full, cv2.COLOR_BGR2RGB)  # Convert to RGB for visualization
plt.imshow(full)
plt.title("Training Image")
plt.show()
```


    
![png](output_1_0.png)
    



```python
test = cv2.imread('Sunflower_Testing.jpg')
test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)  # Convert to RGB for visualization
plt.imshow(test)
plt.title("Test Image")
plt.show()
```


    
![png](output_2_0.png)
    



```python
# Print image shapes
print('Test image shape: ', test.shape)
```

    Test image shape:  (613, 920, 3)
    


```python
print('Training image shape: ', full.shape)
```

    Training image shape:  (225, 225, 3)
    


```python
# Methods for template matching
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
```


```python
for m in methods:
    test_copy = test.copy()  # Copy test image to draw rectangles on
    method = eval(m)  # Evaluate the method string to the actual OpenCV method
    
    # Apply template matching
    res = cv2.matchTemplate(test_copy, full, method)
    
    # Get the minimum and maximum values of the result
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # Choose the top left corner based on the method
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc

    # Get the dimensions of the template image
    height, width, channels = full.shape
    bottom_right = (top_left[0] + width, top_left[1] + height)

    # Draw a rectangle on the test image to indicate the match
    cv2.rectangle(test_copy, top_left, bottom_right, (255, 0, 0), 10)  # Red rectangle

    # Plot the result: heatmap of the matching and the test image with detection
    plt.figure(figsize=(12, 6))  # Adjust the size for better visualization
    plt.subplot(121)
    plt.imshow(res)  # Heatmap of template matching
    plt.title("Heatmap of Template Matching")
    plt.subplot(122)
    plt.imshow(test_copy)  # Image with detected rectangle
    plt.title('Detection of Template')
    plt.suptitle(f'Method: {m}')  # Display the method name on top of the images
    plt.show()

    print('\n')
```


    
![png](output_6_0.png)
    


    
    
    


    
![png](output_6_2.png)
    


    
    
    


    
![png](output_6_4.png)
    


    
    
    


    
![png](output_6_6.png)
    


    
    
    


    
![png](output_6_8.png)
    


    
    
    


    
![png](output_6_10.png)
    


    
    
    


```python

```




