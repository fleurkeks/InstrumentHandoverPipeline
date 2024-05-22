import os

wrkdir = os.getcwd()

with open(".env", 'w+') as logfile:
    logfile.write(("PROJECT_PATH = '" + wrkdir + "'").replace("\\","/"))


