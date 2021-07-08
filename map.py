import numpy as np
import cv2
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt 
# import json


def freq_number(freq_sequence, freq, mark):
    for i in range(N//2):
        if freq_sequence[i] == freq:
            return i
        elif freq_sequence[i] > freq and mark == 0:
            return i-1
        elif freq_sequence[i] > freq:
            return i


def integrate(function, variable, b_edge, t_edge):
    integral = 0
    for i in range(b_edge, t_edge):
        integral += (function[i] + function[i+1]) / 2 * (variable[i+1] - variable[i])
    return integral 



fl = True


# print(frames)

cap = cv2.VideoCapture('neck.mp4')

if (cap.isOpened()== False):
    print("Error opening video stream or file")
    
    
fps = cap.get(cv2.CAP_PROP_FPS)
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = float(n_frames) / float(fps)
T = duration/n_frames
N = n_frames


print("duration of vid", duration)
print("number of frames", n_frames)
print("time between frames", T)

# мальчик мой, это чтобы всё не приходилось три раза перезапускать, вот этот вот while и if с break
while(cap.isOpened()):
    
    ret, frame = cap.read()
    
    if fl:
        d0 = frame.shape[0]
        d1 = frame.shape[1]
        # массив кадров, первый кадр из нулей
        frames = np.zeros((d0,d1,1))
        print("size of frame", d0, d1)
        
        frame = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY)
        
        for i in range(d0):
            for j in range(d1):
                frames[i][j][0] = frame[i][j]
        
        fl = False
        
        # matrix_of_arrays = [[[0]]*(d1)]*(d0)
        
        # for i in range(d0):
        #     for j in range(d1):
        #         matrix_of_arrays[i][j] = frame[i][j]
                
        k=0

    # да, этот if
    elif ret == True:
        
        # перекраска в серость и прекрепление к больничке
        frame = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY)
        frames = np.append(frames, np.reshape(frame, (d0, d1, 1)), axis = 2)

        # print(frame.shape[0], frame.shape[1])
        
        # for i in range(d0):
        #     for j in range(d1):
        #         matrix_of_arrays[i][j].append((frame[i][j]))
        k+=1
        if k%10 == 0:
            print(k)
        
        # for i in range(d0):
        #     for j in range(d1):
        #         frames[0][i][j] = np.append(frames[0][i][j], frame[i][j])

        # cv2.imshow('Frame',frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    else:
        break

# print(frames[1])

cap.release()
cv2.destroyAllWindows()

print("all frames added into array")


xf = fftfreq(N, T)[:N//2]
bottom_edge = freq_number(xf, 0.5, 0)
top_edge = freq_number(xf, 2, 1)

# print(bottom_edge, top_edge)

new_frame = np.zeros((d0,d1,1))

#преобразование фурье
# print("number of pixels", d0*d1)

for i in range(d0):
    if i % 50 == 0:
        print(i)
    for j in range(d1):
        yf = fft(frames[i][j])
        yf = 2.0 / N * np.abs(yf[0:N//2])
        new_frame[i][j][0] = integrate(yf, xf, bottom_edge + 1, top_edge - 1)


while(1):
    cv2.imshow('Frame',new_frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
cap.release()
cv2.destroyAllWindows()


# cv2.imshow('Frame',new_frame)
# if cv2.waitKey(25) & 0xFF == ord('q'):
#             cap.release()
#             cv2.destroyAllWindows()

# cv2.imwrite("img1.jpg", new_frame)
# print("imj is saved")

# я не согласен, но пока так

# print("number of rep", d0*d1*n_frames)

# frames_f = [[0]*d1]*d0

# for i in range(d0):
#     if i%3==0:
#         print("Эра", i)
#     for j in range(d1):
#         a = frames[1][i][j]
#         for k in range(2, n_frames+1):
#             a = np.append(a, frames[k][i][j])
#         a = fft(a)
#         frames_f[i][j] = a

# matrix_of_arrays_f = [[[0]]*d1]*d0
  
# for i in range(d0):
#     print(i+1, "of", d0)
#     for j in range(d1):
#         matrix_of_arrays_f[i][j] = fft(matrix_of_arrays[i][j])
        
      
print("fft applied on all pixels")
            