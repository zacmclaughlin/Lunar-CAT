import ee 

ee.Initialize()

img = ee.Image('LANDSAT/LT05/C01/T1_SR/LT05_034033_20000913')

print(img)

print(img.getInfo())


img_night = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG').filter(ee.Filter.date('2017-05-01','2017-05-31'))

nighttime = img_night.select('avg_rad')

