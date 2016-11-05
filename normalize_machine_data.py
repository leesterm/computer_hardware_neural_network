#@Leester Mei
#Read data and normalize
data = []
max_myct = -1
max_mmin = -1
max_mmax = -1
max_cach = -1
max_chmin = -1
max_chmax = -1
max_prp = -1
max_erp = -1
import sys 
with open(sys.argv[1],'r') as f:
  for line in f:
    d = str.split(line,",")
    if float(d[2]) > max_myct:
      max_myct = float(d[2])
    if float(d[3]) > max_mmin:
      max_mmin = float(d[3])
    if float(d[4]) > max_mmax:
      max_mmax = float(d[4])
    if float(d[5]) > max_cach:
      max_cach = float(d[5])
    if float(d[6]) > max_chmin:
      max_chmin = float(d[6])
    if float(d[7]) > max_chmax:
      max_chmax = float(d[7])
    if float(d[8]) > max_prp:
      max_prp = float(d[8])
    if float(d[9]) > max_erp:
      max_erp = float(d[9])
    data.append([float(d[2]),float(d[3]),float(d[4]),float(d[5]),float(d[6]),float(d[7]),float(d[8]),float(d[9])])

output = open(sys.argv[2],"w")
for i in range(len(data)):
  for j in range(len(data[i])):
    if j == 0:
      data[i][j] = data[i][j]/max_myct
    if j == 1:
      data[i][j] = data[i][j]/max_mmin
    if j == 2:
      data[i][j] = data[i][j]/max_mmax
    if j == 3:
      data[i][j] = data[i][j]/max_cach
    if j == 4:
      data[i][j] = data[i][j]/max_chmin
    if j == 5:
      data[i][j] = data[i][j]/max_chmax
    if j == 6:
      data[i][j] = data[i][j]/max_prp
    if j == 7:
      data[i][j] = data[i][j]/max_erp
  output.write("{},{},{},{},{},{},{},{}\n".format(data[i][0],data[i][1],data[i][2],data[i][3],data[i][4],data[i][5],data[i][6],data[i][7]))
print data