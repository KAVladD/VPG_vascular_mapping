import numpy as np
import cv2
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt 
# import json


fl = True


# print(frames)

cap = cv2.VideoCapture('neck.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = float(n_frames) / float(fps)

if (cap.isOpened()== False):
    print("Error opening video stream or file")

print("duration of vid", duration)

# мальчик мой, это чтобы всё не приходилось три раза перезапускать, вот этот вот while и if с break
while(cap.isOpened()):
    
    ret, frame = cap.read()
    
    if fl:
        d0 = frame.shape[0]
        d1 = frame.shape[1]
        # массив кадров, первый кадр из нулей
        frames = np.zeros((1,d0,d1))
        print("size of frame", d0, d1)
        
        frame = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY)
        
        for i in range(d0):
            for j in range(d1):
                frames[0][i][j] = frame[i][j]
        
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
        frames = np.append(frames, np.reshape(frame, (1, d0, d1)), axis = 0)

        # print(frame.shape[0], frame.shape[1])
        
        # for i in range(d0):
        #     for j in range(d1):
        #         matrix_of_arrays[i][j].append((frame[i][j]))
        k+=1
        print(k)
        
        # for i in range(d0):
        #     for j in range(d1):
        #         frames[0][i][j] = np.append(frames[0][i][j], frame[i][j])

        #cv2.imshow('Frame',frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    else:
        break

# print(frames[1])

cap.release()
cv2.destroyAllWindows()

print("all frames added into array")

#преобразование фурье

T = duration/n_frames

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
            