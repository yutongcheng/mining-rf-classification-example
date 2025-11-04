// After extracting open-pit locations from Cheng et al. (2025), we used the illustrative open-pit locations (Example_Data_1) to obtain ASTER band ratios (mineral indices; Geoscience Australia, 2004) for preparing the subsequent commodity classification.
// As this example contains only a few data points for demonstration purposes, we applied a 15 m buffer around each illustrative open-pit points to include more pixels for the subsequent commodity classification. Please note that the original study used only point data, with one point/pixel per mining site/polygon. 

var Example_Data_1 = ee.FeatureCollection("projects/ee-chengyu-tongb2/assets/OP_training_example/Example_Data_3_15m");// Note: Example_Data_3_15m is the correct dataset used. The variable name (Example_Data_1) is used here because this dataset now appears first, and the data were not re-uploaded.

// Dates of ASTER scenes used
var startDate = '2003-01-01';
var endDate = '2005-12-31';

// Create the ASTER image collection
var collection = createAsterCollection(ee.Filter.date(startDate, endDate));

function createAsterCollection(filters) {
  return ee.ImageCollection('ASTER/AST_L1T_003')
    .filter(ee.Filter.and(
      filters,
      ee.Filter.listContains('ORIGINAL_BANDS_PRESENT', 'B01'),
      ee.Filter.listContains('ORIGINAL_BANDS_PRESENT', 'B02'),
      ee.Filter.listContains('ORIGINAL_BANDS_PRESENT', 'B3N'),
      ee.Filter.listContains('ORIGINAL_BANDS_PRESENT', 'B04'),
      ee.Filter.listContains('ORIGINAL_BANDS_PRESENT', 'B05'),
      ee.Filter.listContains('ORIGINAL_BANDS_PRESENT', 'B06'),
      ee.Filter.listContains('ORIGINAL_BANDS_PRESENT', 'B07'),
      ee.Filter.listContains('ORIGINAL_BANDS_PRESENT', 'B08'),
      ee.Filter.listContains('ORIGINAL_BANDS_PRESENT', 'B09'),
      ee.Filter.listContains('ORIGINAL_BANDS_PRESENT', 'B10'),
      ee.Filter.listContains('ORIGINAL_BANDS_PRESENT', 'B11'),
      ee.Filter.listContains('ORIGINAL_BANDS_PRESENT', 'B12'),
      ee.Filter.listContains('ORIGINAL_BANDS_PRESENT', 'B13'),
      ee.Filter.listContains('ORIGINAL_BANDS_PRESENT', 'B14')
    ))
    .map(preProcess);
}

function preProcess(image) {
  return compose(image, [
    radiance,
    surfaceTemperatureCalc,
    reflectanceCalc,
    mask,
    retrieveCloudMask,
    scale
  ]);
}

function compose(image, operations) {
  return operations.reduce(
    function(image, operation) {
      return image.select([]).addBands(operation(image));
    },
    image
  );
}

function radiance(image) {
  var coefficients = ee
    .ImageCollection(
      image.bandNames().map(function (band) {
        return ee.Image(image.getNumber(ee.String('GAIN_COEFFICIENT_').cat(band))).float();
      })
    )
    .toBands()
    .rename(image.bandNames());
  var radiance_done = image.subtract(1).multiply(coefficients);
  return image
    .addBands(radiance_done, null, true)
    .select(
      ['B01', 'B02', 'B3N', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B10', 'B11', 'B12', 'B13', 'B14'],
      ['green', 'red', 'nir', 'swir1', 'swir2', 'swir3', 'swir4', 'swir5', 'swir6', 'thermal1', 'thermal2', 'thermal3', 'thermal4', 'thermal5']
    );
} 

function surfaceTemperatureCalc(image) {
  var k1 = [3040.136402, 2482.375199, 1935.060183, 866.468575, 641.326517];
  var k2 = [1735.337945, 1666.398761, 1585.420044, 1350.069147, 1271.221673];
  var surfaceTemperature = image
    .select(['thermal1', 'thermal2', 'thermal3', 'thermal4', 'thermal5'])
    .pow(-1)
    .multiply(k1)
    .add(1)
    .log()
    .pow(-1)
    .multiply(k2);
  return image
   .addBands(surfaceTemperature, null, true)
}

function reflectanceCalc(image) {
  var dayOfYear = image.date().getRelative('day', 'year');
  var earthSunDistance = ee.Image().expression(
    '1 - 0.01672 * cos(0.01720209895 * (dayOfYear - 4))',
    {dayOfYear: dayOfYear}
  );
  var sunElevation = image.getNumber('SOLAR_ELEVATION');
  var sunZen = ee.Image().expression(
    '(90 - sunElevation) * pi/180',
    {sunElevation: sunElevation, pi: Math.PI}
  );
  var reflectanceFactor = ee.Image().expression(
    'pi * pow(earthSunDistance, 2) / cos(sunZen)',
    {earthSunDistance: earthSunDistance, sunZen: sunZen, pi: Math.PI}
  );

  // Solar Spectral Irradiance
  var irradiance = [1847, 1553, 1118, 232.5, 80.32, 74.92, 69.20, 59.82, 57.32];
  var reflectance = image
    .select(['green', 'red', 'nir', 'swir1', 'swir2', 'swir3', 'swir4', 'swir5', 'swir6'])
    .multiply(reflectanceFactor)
    .divide(irradiance);
  return image
  .addBands(reflectance, null, true)
}

function mask(image) {
  // Water Mask
  var monthWater = ee.ImageCollection("JRC/GSW1_4/MonthlyHistory");
  var m = image.date().get('month');
  var y = image.date().get('year');
  var water = monthWater.filter(ee.Filter.eq('month', m))
                     .filter(ee.Filter.eq('year', y))
                     .mode();
  var waterMask = water.neq(2);
    
  // ASTER NDSI Calculation
  var ndsi = image.expression('(i.green - i.swir1) / (i.green + i.swir1)',{i: image}).rename('ndsi');
  // Snow Mask
  var snowMask = ndsi.lt(0.4);
  
    var ndvi = ee.Image().expression('(i.nir - i.red) / (i.nir + i.red)', {i: image})
  var vegetationMask = ndvi.lt(0.5)
  
  // Update mask based on water and cloud masks
  var finalMask = image.mask()
                  .and(waterMask)
                  .and(snowMask)

  return image.updateMask(finalMask);
}
  
function retrieveCloudMask(image) {
  // Read visual near infrared (VNIR) channels at 15m resolution.
  var r1 = image.select('red');
  var r2 = image.select('green');
  var r3N = image.select('nir');

  // Read short-wave infrared (SWIR) channels at 30m resolution and match VNIR resolution.
  var r5 = image.select('swir2');

  // Ratios for clear-cloudy-tests.
  var r3N2 = r3N.divide(r2);
  var r12 = r1.divide(r2);

  // Set cloud mask to default "confidently clear".
  var CONFIDENTLY_CLOUDY = 5
  var PROBABLY_CLOUDY = 4
  var PROBABLY_CLEAR = 3
  var CONFIDENTLY_CLEAR = 2

  var clmask = ee.Image.constant(CONFIDENTLY_CLEAR);

  // Cloud mask thresholds
  var probablyClear = r3N.gt(0.03).and(r5.gt(0.01)).and(r3N2.gt(0.7)).and(r3N2.lt(1.75)).and(r12.lt(1.45));
  var probablyCloudy = r3N.gt(0.03).and(r5.gt(0.015)).and(r3N2.gt(0.75)).and(r3N2.lt(1.75)).and(r12.lt(1.35));
  var confidentlyCloudy = r3N.gt(0.065).and(r5.gt(0.02)).and(r3N2.gt(0.8)).and(r3N2.lt(1.75)).and(r12.lt(1.2));

  clmask = clmask.where(probablyClear, PROBABLY_CLEAR);
  clmask = clmask.where(probablyCloudy, PROBABLY_CLOUDY);
  clmask = clmask.where(confidentlyCloudy, CONFIDENTLY_CLOUDY);

  var cloudMask = clmask.lt(5); // Threshold for cloud probability
  
  // Apply cloud mask to the image
  return image.updateMask(cloudMask);
}

function scale(image) {
  return image.multiply(10000);
}

function addRatios(image) {
  var red = image.select('red');
  var green = image.select('green');
  var nir = image.select('nir');
  var swir1 = image.select('swir1');
  var swir2 = image.select('swir2');
  var swir3 = image.select('swir3');
  var swir4 = image.select('swir4');
  var swir5 = image.select('swir5');
  var swir6 = image.select('swir6');
  var thermal1 = image.select('thermal1');
  var thermal2 = image.select('thermal2');
  var thermal3 = image.select('thermal3');
  var thermal4 = image.select('thermal4');
  var thermal5 = image.select('thermal5');

// Add the specified ratio bands with the requested names (Geoscience Australia, 2004)
  var indicator1 = red.divide(green).rename('indicator1');
  var indicator2 = swir2.divide(nir).add(green.divide(red)).rename('indicator2');
  var indicator3 = swir1.divide(swir2).rename('indicator3');
  var indicator4 = swir1.divide(red).rename('indicator4');
  var indicator5 = swir2.divide(swir1).rename('indicator5');
  var indicator6 = swir1.divide(nir).rename('indicator6');
  var indicator7 = (swir4.add(swir6)).divide(swir5).rename('indicator7');
  var indicator8 = (swir3.add(swir6)).divide(swir4.add(swir5)).rename('indicator8');
  var indicator9 = (swir3.add(swir6)).divide(swir5).rename('indicator9');
  var indicator10 = swir3.divide(swir5).rename('indicator10');
  var indicator11 = (swir3.add(swir5)).divide(swir4).rename('indicator11');
  var indicator12 = thermal4.divide(thermal5).rename('indicator12');
  var indicator13 = (swir2.add(swir4)).divide(swir3).rename('indicator13');
  var indicator14 = (swir1.add(swir3)).divide(swir2).rename('indicator14');
  var indicator15 = swir2.divide(swir3).rename('indicator15');
  var indicator16 = swir4.divide(swir3).rename('indicator16');
  var indicator17 = swir4.divide(swir2).rename('indicator17');  
  var indicator18 = (swir2.multiply(swir4)).divide(swir3.multiply(swir3)).rename('indicator18');  
  //var indicator19 = swir1.divide(swir2).rename('indicator19'); 
  //var indicator20 = swir2.divide(swir3).rename('indicator20'); Same ratio as Indicator 15.
  var indicator21 = thermal5.divide(thermal3).rename('indicator21');  
  var indicator22 = (thermal2.multiply(thermal2)).divide(thermal1).divide(thermal3).rename('indicator22');
  var indicator23 = thermal3.divide(thermal4).rename('indicator23');
  //var indicator24 = thermal4.divide(thermal3).rename('indicator24'); Same function as indicator 21 (Geoscience Australia, 2004)
  //var indicator25 = thermal3.divide(thermal4).rename('indicator25'); Same ratio as Indicator 13.  
  var indicator26 = (thermal2.multiply(thermal2)).divide((thermal1.multiply(thermal3))).rename('indicator26');
  var indicator27 = thermal2.divide(thermal1).rename('indicator27'); 
  var indicator28 = thermal2.divide(thermal3).rename('indicator28');   
  var indicator29 = thermal4.divide(thermal1).rename('indicator29'); 
  
  return image.addBands([
    indicator1,
    indicator2,
    indicator3,
    indicator4,
    indicator5,
    indicator6,
    indicator7,
    indicator8,
    indicator9,
    indicator10,
    indicator11,
    indicator12,
    indicator13,
    indicator14,
    indicator15,
    indicator16,
    indicator17,
    indicator18,
    //indicator19,
    //indicator20,
    indicator21,
    indicator22,
    indicator23,
    //indicator24,
    //indicator25,
    indicator26,
    indicator27,
    indicator28,
    indicator29,
  ]);
}

// Map the addRatios function over the collection
var collectionWithRatios = collection.map(addRatios);
var Final_sample = collectionWithRatios.min();
// Specify the bands for training
var bands = ['indicator1','indicator2','indicator3','indicator4', 'indicator5','indicator6',
             'indicator7','indicator8','indicator9','indicator10','indicator11','indicator12',
             'indicator13','indicator14','indicator15','indicator16','indicator17','indicator18','indicator19',
             'indicator21','indicator22','indicator23','indicator26','indicator27','indicator28','indicator29'
             ]

var total_sample = Final_sample.select(bands).sampleRegions({
  collection: Example_Data_1,
  properties:['polygon_id','Commodity','index'],
  scale:15,
  tileScale: 16
})

// Export the scenes to Google Drive as GeoJSON
Export.table.toDrive({
  collection: total_sample,
  description: 'Example_Data_2_pre', // Description for the exported file
  folder: 'YOUR_FOLDER_NAME', // Specify the folder in your Google Drive
  fileFormat: 'CSV' // Export format
});

// Both labelled and unlabelled polygons need to obtain ASTER pixels with band ratios, serving as training and prediction datasets, respectively, in the subsequent commodity classification model.  
// After obtaining ASTER pixels with band ratios, the commodity classification will be conducted using R in RStudio, as it provides advanced statistical and machine learning tools for classification and analysis.
// In the actual workflow, Example_Data_2_pre is processed and then imported into RStudio.

// For HISUI hyperspectral data, we evaluated multiple combinations of indices across the full 185 spectral bands, and the most effective configuration was obtained by averaging bands to mimic ASTER mineral indices 
// These HISUI-derived indices were then combined with ASTER thermal indices (since HISUI lacks thermal bands) to ensure completeness of the spectral features used in the model.
// To further utilise HISUI’s hyperspectral characteristics, which provide finer spectral discrimination and improved sensitivity to subtle mineralogical variations, all HISUI bands were initially included in the classification together with the indices. 
// However, to minimise redundancy and improve model efficiency, a feature-importance analysis was conducted, retaining only the top 25% of features with pairwise correlations below 0.9 for the final model.
// Note: HISUI data are not fully open-access; data usage requires registration and permission from the provider, and are therefore not included in this example.




