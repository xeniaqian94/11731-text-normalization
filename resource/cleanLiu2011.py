'''
    File name: cleanHan2011.py
    Author: xin
    Date created: 4/29/17 11:16 PM
'''

f=open("Liu2011.txt","r")
f_write=open("Liu2011_clean.txt","w")
for line in f.readlines():
    f_write.write(line.split("\t")[1].split("|")[0].strip()+"\t"+line.split("\t")[1].split("|")[1].strip()+"\n")
f_write.close()
f.close()


# sed "s/[[:space:]]\+/ /g" Han2011.txt > Han2011_clean.txt
