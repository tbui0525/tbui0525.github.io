```python
import netCDF4 as nc
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.linalg as la
import scipy
import math
from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import os
os.environ["PROJ_LIB"] = "C:\\Utilities\\Python\\Anaconda\\Library\\share"; #fixr
from mpl_toolkits.basemap import Basemap
```

# Downloading Data


```python
#for year in years:
#   url = 'https://downloads.psl.noaa.gov/Datasets/godas/pottmp.'+str(year)+'.nc'
#    r = requests.get(url, allow_redirects = True)
#    open('pottmp.'+str(year)+'.nc', 'wb').write(r.content)
```

# Compiling and Reformatting Datasets


```python
file = 'C:/Users/tbui0/Downloads/air.mon.mean.nc'
AirData = nc.Dataset(file)

```


```python
years = np.arange(1980,2013)
WaterData = []
for year in years:
    file = 'C:/Users/tbui0/Yearly Pottmps/pottmp.'+str(year)+'.nc'
    WaterData.append(nc.Dataset(file))

```


```python
lats = AirData.variables['lat'][:]
lons = AirData.variables['lon'][:]
time = AirData.variables['time'][:]
air = AirData.variables['air'][:]
levels = AirData.variables['level'][:]
```


```python
lons
```




    masked_array(data=[  0.,   2.,   4.,   6.,   8.,  10.,  12.,  14.,  16.,
                        18.,  20.,  22.,  24.,  26.,  28.,  30.,  32.,  34.,
                        36.,  38.,  40.,  42.,  44.,  46.,  48.,  50.,  52.,
                        54.,  56.,  58.,  60.,  62.,  64.,  66.,  68.,  70.,
                        72.,  74.,  76.,  78.,  80.,  82.,  84.,  86.,  88.,
                        90.,  92.,  94.,  96.,  98., 100., 102., 104., 106.,
                       108., 110., 112., 114., 116., 118., 120., 122., 124.,
                       126., 128., 130., 132., 134., 136., 138., 140., 142.,
                       144., 146., 148., 150., 152., 154., 156., 158., 160.,
                       162., 164., 166., 168., 170., 172., 174., 176., 178.,
                       180., 182., 184., 186., 188., 190., 192., 194., 196.,
                       198., 200., 202., 204., 206., 208., 210., 212., 214.,
                       216., 218., 220., 222., 224., 226., 228., 230., 232.,
                       234., 236., 238., 240., 242., 244., 246., 248., 250.,
                       252., 254., 256., 258., 260., 262., 264., 266., 268.,
                       270., 272., 274., 276., 278., 280., 282., 284., 286.,
                       288., 290., 292., 294., 296., 298., 300., 302., 304.,
                       306., 308., 310., 312., 314., 316., 318., 320., 322.,
                       324., 326., 328., 330., 332., 334., 336., 338., 340.,
                       342., 344., 346., 348., 350., 352., 354., 356., 358.],
                 mask=False,
           fill_value=1e+20,
                dtype=float32)




```python
air = air[(1980-1871)*12::12, :,:,:]
```


```python
CombinedData = []
for g in range(0, len(WaterData)):
    YEARLY_AIR = np.array(air[g].flatten()).tolist()
    ANNUAL_WATER = WaterData[g].variables['pottmp'][:]
    YEARLY_WATER = np.array(ANNUAL_WATER[0].flatten()).tolist()
    CombinedData.append(YEARLY_AIR + YEARLY_WATER)
CombinedData = np.array(CombinedData)
CombinedData.shape
```




    (33, 6412320)




```python
360*418*40+91*180*24
```




    6412320




```python
CombinedData = CombinedData.T
```


```python
#Replaces with Nan because of how GODAS data is formatted
for i in range(0, CombinedData.shape[0]):
    for j in range(0, CombinedData.shape[1]):
        if (CombinedData[i,j] < 0):
            CombinedData[i,j] = np.nan
```


```python
np.save('JanuaryRaw.npy', CombinedData)
```


```python
CombinedData = np.load('JanuaryRaw.npy')
```

# Basic Computations (anomalies, climatology, Standard Deviation, etc)


```python
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    clim = np.nanmean(CombinedData, axis=1)
    sdev = np.nanstd(CombinedData, axis = 1)
```


```python
clim = np.reshape(clim, (clim.size, 1))
sdev = np.reshape(sdev, (sdev.size,1))
```


```python
waterlons = np.array(WaterData[0].variables['lon'][:])
waterlats = np.array(WaterData[0].variables['lat'][:])
xxwater, yywater = np.meshgrid(waterlons, waterlats)
xxair, yyair = np.meshgrid(lons, lats)
depths =WaterData[0].variables['level'][:]
depths.size
```




    40




```python
depths
```




    masked_array(data=[   5.,   15.,   25.,   35.,   45.,   55.,   65.,   75.,
                         85.,   95.,  105.,  115.,  125.,  135.,  145.,  155.,
                        165.,  175.,  185.,  195.,  205.,  215.,  225.,  238.,
                        262.,  303.,  366.,  459.,  584.,  747.,  949., 1193.,
                       1479., 1807., 2174., 2579., 3016., 3483., 3972., 4478.],
                 mask=False,
           fill_value=1e+20,
                dtype=float32)




```python
def f(aw, p, m):
    if aw == 1:
        plt.figure(figsize=(10,4))
        plt.contourf(xxwater,yywater, np.reshape(clim[air[0].size+360*418*m:air[0].size+360*418*(m+1),0],(418,360)),1000, cmap= 'twilight_shifted')
        plt.title('Climatology at Depth ' + str(depths[m]) + ' meters below Sea Level')
        plt.colorbar()
        plt.show()
    else:
        plt.figure(figsize=(10,5))
        plt.contourf(xxair,yyair, np.reshape(clim[91*180*p:91*180*(p+1),0],(91,180)),1000, cmap= 'twilight_shifted')
        plt.title('Climatology at ' + str(levels[p]) + ' hPa')
        plt.colorbar()
        plt.show()
interactive(f, aw= (0,1), p = (0, levels.size), m = (0,depths.size))
```


    interactive(children=(IntSlider(value=0, description='aw', max=1), IntSlider(value=12, description='p', max=24…



```python
anomalies = CombinedData - clim
```


```python
np.save('JanuaryRawAnomalies.npy', anomalies)

```


```python
anom = np.load('JanuaryRawAnomalies.npy')
```


```python
def f(aw, year, m, p):
    if aw == 1:
        plt.figure(figsize=(10,4))
        plt.contourf(xxwater,yywater, np.reshape(anom[air[0].size+360*418*m:air[0].size+360*418*(m+1),year-1980],(418,360)),1000, cmap= 'jet')
        plt.colorbar()
        plt.title('January Anomalies at '+ str(depths[m])+ ' m '+  str(year))
    else: 
        plt.figure(figsize=(10,5))
        plt.contourf(xxair,yyair, np.reshape(anom[91*180*p:91*180*(p+1),year-1980],(91,180)),1000, cmap= 'jet')
        plt.colorbar()
        plt.title('January Anomalies at '+  str(levels[p])+ ' hPa '+ str(year))
interactive(f, aw = (0,1), year = (1980,2012), m = (0, depths.size), p = (0, levels.size))
```


    interactive(children=(IntSlider(value=0, description='aw', max=1), IntSlider(value=1996, description='year', m…



```python
#standardized anomalies
np.seterr(divide='ignore', invalid='ignore')
stnd_anom = (CombinedData - clim)/ sdev
```


```python
np.save('StandardizedCombined.npy', stnd_anom)
```


```python
stnd_anom = np.load('StandardizedCombined.npy')

```


```python
def f(aw, year, m, p):
    if aw == 1:
        plt.figure(figsize=(10,4))
        plt.contourf(xxwater,yywater, np.reshape(StdData[air[0].size+360*418*m:air[0].size+360*418*(m+1),year-1980],(418,360)),1000, cmap= 'jet')
        plt.colorbar()
        plt.title('January Standardized Anomalies at '+ str(depths[m])+ ' m '+  str(year))
    else: 
        plt.figure(figsize=(10,5))
        plt.contourf(xxair,yyair, np.reshape(StdData[91*180*p:91*180*(p+1),year-1980],(91,180)),1000, cmap= 'jet')
        plt.colorbar()
        plt.title('January Standardized Anomalies at '+  str(levels[p])+ ' hPa '+ str(year))
interactive(f, aw = (0,1), year = (1980,2012), m = (0, depths.size), p = (0, levels.size))
```


    interactive(children=(IntSlider(value=0, description='aw', max=1), IntSlider(value=1996, description='year', m…


# Weight Computations

$$ \sqrt{c_p \rho \Delta d_{ij}  \cos{\phi}\Delta \theta \Delta \phi }$$


```python
lats4air = np.array(yyair).flatten().tolist()
lats4water = np.array(yywater).flatten().tolist()
```


```python
len(lats4air + lats4water)
```




    166860




```python
lats4weighing = []
for i in range(levels.size):
    lats4weighing += lats4air
for j in range(depths.size):
    lats4weighing += lats4water
```


```python
lats4weighing = np.array(lats4weighing)
lats4weighing.shape
```




    (6412320,)




```python
#cos phi
weightedA = np.sqrt(np.cos(lats4weighing*np.pi/180))

```


```python
weightedA = np.reshape(weightedA, (weightedA.size, 1))
weightedA.shape
```




    (6412320, 1)




```python
weightedAnom = stnd_anom * weightedA
```


```python
airclim = clim[0:air[0].size]
airclim.size
```




    393120



 Height calculations
 $$ P = P_0 e^{\frac{-gM(h-h_0)}{RT}}$$


```python
r = 8.31432
g = -9.8
m = 0.0289644
```


```python
airheights= []
for i in range(airclim.size):
    airheights.append(airclim[i]*r * math.log(levels[int(i/(91*180))]/1013.25)/(g*m))
airheights = np.array(airheights)
```


```python
np.save('air height.npy', airheights)
```


```python
airheights = np.load('air height.npy')
```

Thickness 
$$ \Delta d_{ij}$$


```python
waterthickness = [5]
for i in range(0,depths.size-1):
    waterthickness.append(depths[i+1]-depths[i])
waterthickness = np.array(waterthickness)
waterthickness = np.reshape(waterthickness, (waterthickness.size, 1))
```


```python
thickness = np.empty([stnd_anom.shape[0], 1])
thickness[0:91*180] = airheights[0:91*180]
for i in range(91*180, air[0].size):
    thickness[i] = airheights[i] - airheights[i-91*180]
for j in range(air[0].size, stnd_anom.shape[0]):
    thickness[j] = waterthickness[int((j-air[0].size)/(360*418))]
```

$$ \Delta \theta \Delta \phi$$


```python
airgrid = 4
watergrid = 1/3
```


```python
gridSize = np.empty([weightedAnom.shape[0], 1])
for i in range(0, air[0].size):
    gridSize[i] = airgrid
for j in range(air[0].size, gridSize.size):
    gridSize[j] = watergrid
```


```python
gridSize.size
```




    6412320



Air and Ocean Density


```python
waterdensity = 1000
density = np.empty([weightedAnom.shape[0], 1])
```


```python
for i in range(0,air[0].size):
    density[i] =  levels[int(i/(91*180))]/(2.869*clim[i])
for j in range(air[0].size,density.size):
    density[j] = waterdensity
```


```python
#Heat capacity
cpair = 1.005
cpwater = 4.812
```


```python
heatCap = np.empty([weightedAnom.shape[0], 1])
for i in range(0, air[0].size):
    heatCap[i] = cpair
for j in range(air[0].size, gridSize.size):
    heatCap[j] = cpwater
```


```python
vweightedanom = weightedAnom * np.sqrt(thickness) * np.sqrt(density) * np.sqrt(heatCap) * np.sqrt(gridSize)
```


```python
np.save('Fully Weighted Anomalies.npy',vweightedanom)
```


```python
matslessanom= vweightedanom/np.sqrt(heatCap)
matslessanom /= np.sqrt(density)
```


```python
vweightedanom[air[0].size+170]
```




    array([ 55.74235885,  47.80202568,  12.0146086 ,  53.05830257,
            20.51412016,  23.75735483,  19.06025634,  39.19067845,
            20.62595583,  12.23827996,  41.87473473,  40.30903523,
            28.23078197,  25.88223272, -27.46562259,  11.67910156,
             8.8832096 ,  55.63052317,  10.33707342,   0.83104076,
            28.23078197,  27.0005895 ,  37.84865031, -43.23445324,
            37.96048599, -65.82526028, -63.36487535, -81.70816332,
           -76.78515676, -78.01534922, -64.1477251 , -78.1271849 ,
           -80.02839143])




```python
matslessanom[air[0].size+170]
```




    array([ 0.80356781,  0.68910197,  0.17319957,  0.76487513,  0.29572639,
            0.34248005,  0.27476786,  0.56496296,  0.29733859,  0.17642396,
            0.60365564,  0.58108491,  0.40696785,  0.37311175, -0.3959375 ,
            0.16836299,  0.12805811,  0.80195562,  0.14901665,  0.01198007,
            0.40696785,  0.3892337 ,  0.54561662, -0.623257  ,  0.54722881,
           -0.94892038, -0.91345209, -1.17788432, -1.10691549, -1.12464964,
           -0.92473746, -1.12626183, -1.15366915])




```python
def f(aw, year, m, p):
    if aw == 1:
        plt.figure(figsize=(10,4))
        plt.contourf(xxwater,yywater, np.reshape(vweightedanom[air[0].size+360*418*m:air[0].size+360*418*(m+1),year-1980],(418,360)),1000, cmap= 'jet')
        plt.colorbar()
        plt.title('January Weighted Anomalies at '+ str(depths[m])+ ' m '+  str(year))
    else: 
        plt.figure(figsize=(10,5))
        plt.contourf(xxair,yyair, np.reshape(vweightedanom[91*180*p:91*180*(p+1),year-1980],(91,180)),1000, cmap= 'jet')
        plt.colorbar()
        plt.title('January Weighted Anomalies at '+  str(levels[p])+ ' hPa '+ str(year))
interactive(f, aw = (0,1), year = (1980,2012), m = (0, depths.size), p = (0, levels.size))
```


    interactive(children=(IntSlider(value=0, description='aw', max=1), IntSlider(value=1996, description='year', m…


# Cutting off final ocean layers in EOF computation


```python

```




    array([1.35551041])




```python
vweightedanom.shape
```




    (6412320, 33)




```python
depths[33]
```




    1807.0




```python
vweightedAnom = vweightedanom[0:air[0].size+360*418*34,:]
vweightedAnom.shape
```




    (5509440, 33)




```python
matslessAnom = matslessanom[0:air[0].size+360*418*34,:]
matslessAnom.shape
```




    (5509440, 33)



# EOF computation and graphs


```python
df = pd.DataFrame(data = vweightedAnom)
dropna = df.dropna()
NanlessAnom = dropna.to_numpy()
NanlessAnom.shape
```




    (3895864, 33)




```python
Sigma = np.matmul(NanlessAnom.T, NanlessAnom)
eigenvalues, eigenvectors = scipy.linalg.eig(Sigma)
```


```python
eigenvectors = eigenvectors.T
```


```python
index = np.argsort(eigenvalues)[::-1]
eigvals = eigenvalues[index]
eigvecs = eigenvectors[index]

```


```python

num_eval = np.arange(eigvals.shape[0])+1
cumulative_eval = np.cumsum(eigvals)
```


```python
for i in range(eigvecs.shape[0]):
    print(np.linalg.norm(eigvecs[i]))
```

    0.9999999999999999
    0.9999999999999999
    1.0
    1.0
    0.9999999999999999
    0.9999999999999999
    1.0
    0.9999999999999999
    1.0
    1.0
    0.9999999999999999
    1.0000000000000002
    1.0
    0.9999999999999999
    1.0000000000000002
    1.0
    1.0
    1.0000000000000002
    1.0
    0.9999999999999998
    0.9999999999999999
    1.0
    1.0
    0.9999999999999999
    1.0000000000000002
    1.0000000000000002
    1.0
    1.0
    1.0
    1.0
    1.0000000000000002
    1.0
    1.0
    


```python
plt.figure(figsize=(30., 14.))
fig, ax = plt.subplots()

p1, = plt.plot(num_eval,(eigvals/cumulative_eval[-1])*100, 'b', marker = 'o',label = 'Percentage Variance')
ax.set_ylabel("Percentage Variance")
ax.yaxis.label.set_color('blue')
ax.tick_params('y', colors='b')

ax2 = ax.twinx()
p2, = plt.plot(num_eval,(cumulative_eval/cumulative_eval[-1])*100,'r', marker = 'x',label = 'Cumulative Percentage Variance')
ax2.tick_params('y', colors='r')
ax2.set_ylabel("Cumulative Percentage Variance")
ax2.yaxis.label.set_color('red')

plt.legend(handles=[p1,p2],loc='center right')

plt.show()

```

    C:\Users\tbui0\Anaconda3\lib\site-packages\numpy\core\_asarray.py:83: ComplexWarning:
    
    Casting complex values to real discards the imaginary part
    
    C:\Users\tbui0\Anaconda3\lib\site-packages\numpy\core\_asarray.py:83: ComplexWarning:
    
    Casting complex values to real discards the imaginary part
    
    


    <Figure size 2160x1008 with 0 Axes>



    
![png](output_74_2.png)
    



```python
EOFS = []

for j in range(0,stnd_anom.shape[1]):
        EOFS.append(np.matmul(vweightedAnom, eigvecs[j])/np.linalg.norm(np.matmul(NanlessAnom, eigvecs[j])))
EOF1 = np.array(EOFS)
```


```python
EOF1.shape
EOF1 = EOF1.T
EOF1.shape
```




    (5509440, 33)




```python
for i in range(0,33):
    test1 = pd.DataFrame(EOF1[:,i])
    test2 = test1.dropna()
    test3 = test2.to_numpy()
    print(np.linalg.norm(test3))
```

    1.0000000000000004
    1.0000000000000004
    1.0000000000000009
    1.000000000000002
    1.000000000000001
    0.9999999999999997
    1.0000000000000002
    1.0000000000000002
    1.000000000000001
    0.9999999999999998
    0.9999999999999994
    1.0000000000000016
    0.9999999999999991
    1.0000000000000004
    0.9999999999999993
    1.0
    1.0000000000000004
    0.9999999999999993
    1.0
    0.9999999999999988
    0.9999999999999988
    0.9999999999999998
    1.0000000000000007
    0.9999999999999997
    0.9999999999999993
    1.0000000000000007
    0.9999999999999994
    0.9999999999999998
    0.9999999999999999
    1.0000000000000002
    0.9999999999999993
    0.9999999999999981
    1.000000000000003
    


```python
plt.figure(figsize=(10,5))
plt.contourf(xxair,yyair, 0-np.reshape(PhysicalEOFs[91*180*0:91*180*1,0],(91,180)),1000, cmap= 'jet')

plt.colorbar()
```




    <matplotlib.colorbar.Colorbar at 0x216a3842730>




    
![png](output_78_1.png)
    



```python
EOF1.shape
```




    (6412320, 33)




```python
np.save('Air and Ocean EOFs (not physical).npy', EOF1)
```


```python
#Physical EOFs
np.seterr(divide='ignore', invalid='ignore')
PhysicalEOFs = EOF1/(weightedA[0:air[0].size+360*418*34] * np.sqrt(thickness[0:air[0].size+360*418*34]) * np.sqrt(density[0:air[0].size+360*418*34]) * np.sqrt(heatCap[0:air[0].size+360*418*34]) * np.sqrt(gridSize[0:air[0].size+360*418*34]))
```


```python
depths
```




    masked_array(data=[   5.,   15.,   25.,   35.,   45.,   55.,   65.,   75.,
                         85.,   95.,  105.,  115.,  125.,  135.,  145.,  155.,
                        165.,  175.,  185.,  195.,  205.,  215.,  225.,  238.,
                        262.,  303.,  366.,  459.,  584.,  747.,  949., 1193.,
                       1479., 1807., 2174., 2579., 3016., 3483., 3972., 4478.],
                 mask=False,
           fill_value=1e+20,
                dtype=float32)




```python
np.nanmax(PhysicalEOFs)
```




    3.781304663455402e-05




```python
np.nanmin(PhysicalEOFs)
```




    -4.433764042286555e-05




```python
def f(aw, mode, m, p):
    if aw == 1:
        plt.figure(figsize=(10,4))
        plt.contourf(xxwater,yywater, 0-np.reshape(PhysicalEOFs[air[0].size+360*418*m:air[0].size+360*418*(m+1),mode-1],(418,360)),1000, cmap= 'twilight_shifted')
        plt.colorbar()
        plt.title('January Physical EOFs at '+ str(depths[m])+ ' m. Mode '+  str(mode))
    else: 
        ig=plt.figure(figsize=(10, 5) )

        # Miller projection:
        m=Basemap(projection='mill',lat_ts=10,llcrnrlon=lons.min(), \
          urcrnrlon=lons.max(),llcrnrlat=lats.min(),urcrnrlat=lats.max(), \
          resolution='c')

        x, y = m(*np.meshgrid(lons,lats))

        m.pcolormesh(x,y,0-np.reshape(PhysicalEOFs[91*180*p:91*180*(p+1),mode-1],(91,180)),shading='flat',cmap=plt.cm.twilight_shifted)
        m.colorbar(location='right')
        m.drawcoastlines()
        m.drawmapboundary()
        m.drawparallels(np.arange(-90.,90.,30.),labels=[1,0,0,0])
        m.drawmeridians(np.arange(-180.,180.,60.),labels=[0,0,0,1])
        plt.title('January Air Physical EOFs at '+ str(levels[p]) + 'hPa. Mode ' + str(mode) )
interactive(f, aw = (0,1), mode = (1,34), m = (0, 33), p = (0, levels.size))
```


    interactive(children=(IntSlider(value=0, description='aw', max=1), IntSlider(value=17, description='mode', max…



```python
np.save('Air and Ocean Physical EOFs (down to 2000m).npy', PhysicalEOFs)
```


```python
PhysicalEOFs = np.load('Air and Ocean Physical EOFs (down to 2000m).npy')
```


```python
plt.figure(figsize=(10,4))
plt.contourf(xxwater,yywater, 0-np.reshape(PhysicalEOFs[air[0].size+360*418*0:air[0].size+360*418*1,0],(418,360)),1000, cmap= 'jet')
plt.clim(-3*10**-6, 3*10**-6)
plt.colorbar()
```




    <matplotlib.colorbar.Colorbar at 0x143339116a0>




    
![png](output_88_1.png)
    



```python
np.nanmin(PhysicalEOFs[:,0])
```




    -3.4741296594994454e-06




```python
plt.figure(figsize=(10,4))
plt.contourf(xxwater,yywater, 0-np.reshape(PhysicalEOFs[air[0].size+360*418*10:air[0].size+360*418*11,0],(418,360)),1000, cmap= 'twilight_shifted')
plt.clim(-3*10**-6, 3*10**-6)
plt.colorbar()
```




    <matplotlib.colorbar.Colorbar at 0x2d8c42763d0>




    
![png](output_90_1.png)
    



```python
PhysicalEOFs = np.load('AirandOceanPhysicalEOFs.npy')
```

# Calculating EOFs while ignoring Material Properties


```python
matlessdf = pd.DataFrame(data = matslessAnom)
matlessdropna = matlessdf.dropna()
NoNanNoMat = matlessdropna.to_numpy()
NoNanNoMat.shape
```




    (3895864, 33)




```python
MatlessSigma = np.matmul(NoNanNoMat.T, NoNanNoMat)
matlessEigenvalues, matlessEigenvectors = scipy.linalg.eig(MatlessSigma)
```


```python
matlessEigenvectors = matlessEigenvectors.T
```


```python
NoMatindex = np.argsort(matlessEigenvalues)[::-1]
NoMateigvals = matlessEigenvalues[NoMatindex]
NoMateigvecs = matlessEigenvectors[NoMatindex]
```


```python
num_eval = np.arange(eigvals.shape[0])+1
MatlessCum = np.cumsum(NoMateigvals)
```


```python
NoMateigvals
```




    array([ 1.20845270e+10+0.j,  4.86195639e+09+0.j,  3.80471976e+09+0.j,
            3.41073214e+09+0.j,  1.83069027e+09+0.j,  1.56110927e+09+0.j,
            1.26979555e+09+0.j,  1.23162654e+09+0.j, -5.42367917e-05+0.j,
            1.07772381e+09+0.j,  1.00187010e+09+0.j,  9.41700763e+08+0.j,
            9.02922297e+08+0.j,  7.79071943e+08+0.j,  2.14838863e+08+0.j,
            6.99179983e+08+0.j,  2.46949842e+08+0.j,  2.77262006e+08+0.j,
            3.11198713e+08+0.j,  6.50163127e+08+0.j,  3.49199158e+08+0.j,
            6.19824606e+08+0.j,  4.71597088e+08+0.j,  5.67975062e+08+0.j,
            5.37739467e+08+0.j,  5.13480181e+08+0.j,  5.85130823e+08+0.j,
            4.37886547e+08+0.j,  4.29399795e+08+0.j,  3.83894455e+08+0.j,
            3.58841779e+08+0.j,  2.92866595e+08+0.j,  1.16190656e+09+0.j])




```python
plt.figure(figsize=(30., 14.))
fig, ax = plt.subplots()

p1, = plt.plot(num_eval,(NoMateigvals/MatlessCum[-1])*100, 'b', marker = 'o',label = 'Percentage Variance')
ax.set_ylabel("Percentage Variance")
ax.yaxis.label.set_color('blue')
ax.tick_params('y', colors='b')

ax2 = ax.twinx()
p2, = plt.plot(num_eval,(MatlessCum/MatlessCum[-1])*100,'r', marker = 'x',label = 'Cumulative Percentage Variance')
ax2.tick_params('y', colors='r')
ax2.set_ylabel("Cumulative Percentage Variance")
ax2.yaxis.label.set_color('red')

plt.legend(handles=[p1,p2],loc='center right')

plt.show()

```

    C:\Users\tbui0\Anaconda3\lib\site-packages\numpy\core\_asarray.py:83: ComplexWarning:
    
    Casting complex values to real discards the imaginary part
    
    C:\Users\tbui0\Anaconda3\lib\site-packages\numpy\core\_asarray.py:83: ComplexWarning:
    
    Casting complex values to real discards the imaginary part
    
    


    <Figure size 2160x1008 with 0 Axes>



    
![png](output_99_2.png)
    



```python
MatlessEOFs = []
```


```python
MatlessEOFS = []

for j in range(0,stnd_anom.shape[1]):
        MatlessEOFs.append(np.matmul(matslessAnom, NoMateigvecs[j])/np.linalg.norm(np.matmul(NoNanNoMat, NoMateigvecs[j])))
MatlessEOFs = np.array(MatlessEOFs).T
MatlessEOFs.shape
```




    (5509440, 33)




```python
FizikleEOFs = MatlessEOFs/(weightedA[0:air[0].size+360*418*34] * np.sqrt(thickness[0:air[0].size+360*418*34]) * np.sqrt(gridSize[0:air[0].size+360*418*34]))
```


```python
np.nanmax(FizikleEOFs[:,0])
```




    4.242149755232419e-05




```python
np.nanmin(FizikleEOFs[:,0])
```




    -5.14569522221394e-05




```python
plt.figure(figsize=(10,4))
plt.contourf(xxwater,yywater, 0-np.reshape(FizikleEOFs[air[0].size+360*418*0:air[0].size+360*418*1,0],(418,360)),1000, cmap= 'jet')
plt.clim(-3e-5, 3e-5)
plt.colorbar()
```




    <matplotlib.colorbar.Colorbar at 0x21a2b4982e0>




    
![png](output_105_1.png)
    



```python
def f(aw, mode, m, p):
    if aw == 1:
        plt.figure(figsize=(10,4))
        plt.contourf(xxwater,yywater, 0-np.reshape(FizikleEOFs[air[0].size+360*418*m:air[0].size+360*418*(m+1),mode-1],(418,360)),1000,
                     cmap= 'jet')
        plt.colorbar()
        plt.title('January Physical EOFs at '+ str(depths[m])+ ' m. Mode '+  str(mode))
    else: 
        ig=plt.figure(figsize=(10, 5) )

        # Miller projection:
        m=Basemap(projection='mill',lat_ts=10,llcrnrlon=lons.min(), \
          urcrnrlon=lons.max(),llcrnrlat=lats.min(),urcrnrlat=lats.max(), \
          resolution='c')

        x, y = m(*np.meshgrid(lons,lats))

        m.pcolormesh(x,y,0-np.reshape(FizikleEOFs[91*180*p:91*180*(p+1),mode-1],(91,180)),shading='flat',cmap=plt.cm.jet)
        m.colorbar(location='right')
        m.drawcoastlines()
        m.drawmapboundary()
        m.drawparallels(np.arange(-90.,90.,30.),labels=[1,0,0,0])
        m.drawmeridians(np.arange(-180.,180.,60.),labels=[0,0,0,1])
        plt.title('January Air Physical EOFs at '+ str(levels[p]) + 'hPa. Mode ' + str(mode) )
interactive(f, aw = (0,1), mode = (1,34), m = (0, 33), p = (0, levels.size))
```


    interactive(children=(IntSlider(value=0, description='aw', max=1), IntSlider(value=17, description='mode', max…

