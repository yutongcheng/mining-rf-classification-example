var Example_Data_1 = ee.FeatureCollection("projects/ee-chengyu-tongb2/assets/OP_training_example/Example_Data_1");
var Example_Data_2 = ee.FeatureCollection("projects/ee-chengyu-tongb2/assets/OP_training_example/Example_Data_2");
var Example_Data_1 = Example_Data_1.sort('index');

// Dates of ASTER scenes used for training the model
var startDate = '2003-01-01';
var endDate = '2005-12-31';

// Create the Aster image collection
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

// preProcess  
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

function addAllbands(image) {
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

  return image.addBands([
    red,
    green,
    nir,
    swir1,
    swir2,
    swir3,
    swir4,
    swir5,
    swir6,
    thermal1,
    thermal2,
    thermal3,
    thermal4,
    thermal5
   ]);
}


// Map the addAllbands function over the collection
var collectionWithAllbands = collection.map(addAllbands);
var Final_sample = collectionWithAllbands.min();
// Specify the bands for training
var bands = [    
    'red',
    'green',
    'nir',
    'swir1',
    'swir2',
    'swir3',
    'swir4',
    'swir5',
    'swir6',
    'thermal1',
    'thermal2',
    'thermal3',
    'thermal4',
    'thermal5'
    ]

// Extracts training samples from the Final_sample image using Example_Data_1 as reference points. 
// Each sample is assigned attributes from Example_Data_1, including 'Label' (class identifier), 
// 'Commodity', and 'index'. 
// Sampling is performed at a spatial resolution of 15 meters, with tile scaling set to 16 to optimize processing.
var total_sample = Final_sample.select(bands).sampleRegions({
  collection: Example_Data_1,
  properties:['Label','Commodity','index'],
  scale:15,
  tileScale: 16
})

// Total number of ASTER scenes used
print(total_sample.size())

// Get the distribution of labels
var labelDistribution = Example_Data_1.aggregate_histogram('Label');
print('Label Distribution:', labelDistribution);

// Add a random column for splitting
var total_sample_with_random = total_sample.sort('index').randomColumn('random', 123);

// Split the data into training and validation sets (80% training, 20% validation)
var split = 0.8;
var trainingSample = total_sample_with_random.filter(ee.Filter.lt('random', split));
var validationSample = total_sample_with_random.filter(ee.Filter.gte('random', split));

// Train the random forest model
var classifier = ee.Classifier.smileRandomForest(800).setOutputMode('MULTIPROBABILITY').train({
  features: trainingSample,
  classProperty: 'Label',
  inputProperties: bands
});

// Since the model outputs probabilities instead of discrete classifications, the confusion matrix and accuracy cannot be computed directly. However, if the classification output is generated instead of probabilities, the following codes can be used to evaluate model performance:
// var validationClassification = validationSample.classify(classifier);
// var validationAccuracy = validationClassification.errorMatrix('Label', 'classification');
// print('Validation accuracy:', validationAccuracy.accuracy());
// print('Confusion matrix:', validationAccuracy);

// Dates of ASTER scenes used for identifying open pits (prediction)
var startDate2 = '2003-01-01';
var endDate = '2005-12-31';

// Create the ASTER image collection for prediction on unknown (non-training) data.  
// This dataset is used for identifying open pits in areas not included in the training phase.  
var collection2 = createAsterCollection(ee.Filter.date(startDate2, endDate));

// Map the addAllbands function over the collection
var collectionWithAllbands2 = collection2.map(addAllbands);
var new_sample_pre = collectionWithAllbands2.min();

// Extract feature samples from the processed ASTER image (new_sample_pre) using Example_Data_2 (500-metre grid centroids), which contains locations for open pit identification in unknown (non-training) areas.
var new_sample = new_sample_pre.select(bands).sampleRegions({
  collection: Example_Data_2,
  scale: 15,
  tileScale: 16
});

// Apply the trained Random Forest classifier to the new sample dataset (new_sample) 
var NewClassification = new_sample.classify(classifier);

// Export the classified results to Google Drive as CSV
Export.table.toDrive({
  collection: NewClassification,
  description: 'Example_Data_3_pre',
  folder: 'YOUR_FOLDER_NAME', // Replace with your folder name in Google Drive
  fileFormat: 'CSV'
  });
  
// The output, Example_Data_3_pre, in this example consists of a few 500-metre grid centroids, each assigned probabilities for six land-use categories.  
// To systematically analyse unlabelled polygons, we generated 500-metre grid centroids and progressively reduced the grid size to enhance open pit detection.  
// If multiple open pit points were identified within a single polygon, the point with the highest probability was selected as the final data point for using to obtain ASTER scense in GEE.
// After processing Example_Data_3_pre, the open pit locations for each mining polygon/site will be uploaded to GEE to retrieve ASTER scenes with band values for subsequent commodity classification.