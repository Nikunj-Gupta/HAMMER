import os, glob, pprint 


m_files = glob.glob(os.path.realpath("analysis-by-rendering/message/")+"/**/*.avi", recursive=True) 
nm_files = glob.glob(os.path.realpath("analysis-by-rendering/no_message/")+"/**/*.avi", recursive=True) 
if not os.path.exists("analysis-by-rendering/merged"): os.makedirs("analysis-by-rendering/merged")

res=[] 
for f in m_files: 
    temp = [f] 
    x = os.path.basename(os.path.dirname(f)) 
    temp.extend([f2 for f2 in nm_files if x == os.path.basename(os.path.dirname(f2))])
    res.append(temp) 
pprint.pprint(res) 

for i in res: 
    command = "ffmpeg \
        -i {} \
        -i {} \
        -filter_complex '[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]' \
        -map '[vid]'   -c:v libx264   -crf 23 -preset veryfast {}".format(
            i[0], i[1], os.path.join("analysis-by-rendering", "merged", os.path.basename(os.path.dirname(i[0])+".mp4"))) 
    
    os.system(command) 

