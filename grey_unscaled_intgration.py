import numpy as np
import cv2
from scipy.fft import fft, fftfreq


# return number of given frequency
def freq_number(freq_sequence, freq, n, mark):
    for i in range(n//2):
        if freq_sequence[i] == freq:
            return i
        elif freq_sequence[i] > freq and mark == 0:
            return i-1
        elif freq_sequence[i] > freq:
            return i

# unscaled integration by trapezoids
def integrate(function, variable, b_edge, t_edge):
    integral = 0
    for i in range(b_edge, t_edge):
        integral += (function[i] + function[i+1]) / 2
    return integral 

# maximum in array
def maximum(array, dim0, dim1):
    m = 0
    for i in range(dim0):
        for j in range(dim1):
            if m < array[i][j][0]:
                m = array[i][j][0]
    return m


# open mp4
vid = cv2.VideoCapture('neck.mp4')

if (vid.isOpened()== False):
    print("Error opening video stream or file")
    
# vid info
fps = vid.get(cv2.CAP_PROP_FPS)
n_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
duration = float(n_frames) / float(fps)
T = duration/n_frames
N = n_frames

print("duration of vid", duration)
print("number of frames", n_frames)
print("time between frames", T)



fl = True

while(vid.isOpened()):
    
    ret, frame = vid.read()
    
    # initializing arrays of frames
    if fl:
        
        d0 = frame.shape[0]
        d1 = frame.shape[1]
        
        frames = np.zeros((d0,d1,1))
        
        print("size of frame", d0, d1)
        
        # color into grey
        frame = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY)
        
        for i in range(d0):
            for j in range(d1):
                frames[i][j][0] = frame[i][j]
        
        fl = False
        
        k=0
        
        
    elif ret == True:
        
        # color into grey
        frame = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY)
        
        # add frame into array
        frames = np.append(frames, np.reshape(frame, (d0, d1, 1)), axis = 2)

        k+=1
        if k%10 == 0:
            print(k)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    else:
        break

print("all frames added into array")

# frequencys array
xf = fftfreq(N, T)[:N//2]

# thresholds for given frequency
bottom_threshold = freq_number(xf, 0.5, N, 0)
top_threshold = freq_number(xf, 2, N, 1)


# matrix for desired frame
new_frame = np.zeros((d0,d1,1))

# fourier transform for every pixels and filling new_frame
for i in range(d0):
    if i % 50 == 0:
        print(i)
    for j in range(d1):
        yf = fft(frames[i][j])
        yf = 2.0 / N * np.abs(yf[0:N//2])
        new_frame[i][j][0] = integrate(yf, xf, bottom_threshold, top_threshold)


max_ = maximum( new_frame, d0, d1)

for i in range(d0):
    for j in range(d1):
        new_frame[i][j] = new_frame[i][j] / max_


while(1):
    cv2.imshow('Frame',new_frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
vid.release()
cv2.destroyAllWindows()