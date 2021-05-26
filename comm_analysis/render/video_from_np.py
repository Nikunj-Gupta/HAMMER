import numpy as np, cv2, os 
 

def convert(filepath): 
    frames = np.load(filepath) 
    frameSize = (700, 700)

    out = cv2.VideoWriter(os.path.join(os.path.dirname(filepath), 'output_video.avi'), cv2.VideoWriter_fourcc(*'DIVX'), 5, frameSize)

    for i in range(len(frames)):
        img = frames[i] 
        out.write(img)
    out.release()
    print(filepath+" Done!")

where = "analysis-by-rendering/" 
for subdir, dirs, files in os.walk(where): 
    for file in files:
        filepath = subdir + os.sep + file 
        if filepath.endswith(".npy"):
            convert(filepath) 