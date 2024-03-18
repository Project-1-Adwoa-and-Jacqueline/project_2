# Project_2: Ride-With-Us!!!
![](Images\NYC_Rideshare.png)
 
## Motivation && Summary : 

Ride-With-Us project seeks to analyze the rideshare drivers in NYC area to help maximize their earnings by applying:
1.  the Machine Learning Techniques to determine whether a ride will tip or not while optimizing NYC Rideshare Drivers from companies like Juno, Uber, Via, and Lyft,and 
2. a schedule that targets top ~5% fares implementing the Facebook Prophet Model.
3. the best pick-up locations and round-trip routes to target 

Based  on our Machine Learning models, we hope to make drivers' trip worthy by determining whether particular pickup and drop off location will help drivers reach their earning for a prticular day, week, month, on and on. This project retrieved its data from the original source, NYC taxi and Limo Commission, a public domain, in a parquet format. Utilizing previous year's data, 2023, and due to the high volume of datasets and in order to get a manageable dataset for the project, statistical sampling of 1000 dataset each for each month of the year 2023 was used for this project. We opted for the **High Volume For-Hire Vehicle Trip Records (HVFHV)** beacsue it has the necessary columns needed to help with our research. Included in our dataset is a separate file, **Taxi Zone Lookup Table** to help map the locations with the best fares or not so best fares. 

**Historic Data Summary**: 

“TLC Trip Record Data”
* Source: NYC Taxi and Limo Commission
* License: Public Domain
* Format: Parquet (.read_parquet)
* Twelve parquet files:
    * one file for each month of 2023
    * trip data for HVFHS “high-volume for-hire services” (Uber, Lyft, Via, and Juno)
* Other related files:
    * High Volume FHV Trips Data Dictionary (PDF)
    * Taxi Zone Lookup Table (CSV)

Columns:
> Index(['request_datetime', 'hvfhs_license_num', 'dispatching_base_num',
>       'originating_base_num', 'on_scene_datetime', 'pickup_datetime',
>       'dropoff_datetime', 'PULocationID', 'DOLocationID', 'trip_miles',
>       'trip_time', 'base_passenger_fare', 'tolls', 'bcf', 'sales_tax',
>       'congestion_surcharge', 'airport_fee', 'tips', 'driver_pay',
>       'shared_request_flag', 'shared_match_flag', 'access_a_ride_flag',
>       'wav_request_flag', 'wav_match_flag'],
>      dtype='object')

## Model Summaries
Model 1 & 2 focuses on determining the tip worthiness of a ride and Model 3, an application to help determine the best locations drivers can earn the most while achieving their financial and Income goal. Tips and Drivers Income were converted to binary classification to aid with the machine learning process.
* Model 1: Logistic Regression
* Model 2: RandomForest && Neural Network
* Model 3: Facebook Prophet
 * y-hat2: good_fare (bool): top ~5% fares
    * This model enables drivers to forecast high fares to plan their schedules.
    * Reasoning: time series data, seasonality, missing data points (limited sample, rare occurrence)


## Summary Statistics for All Fares
* Average:
    * Total pay per ride: $20.45
    * Fare: $ 19.31
    * Tip: $ 1.14 (only 20% of customers tip!) 
    * Trip length: 5 miles
    * Trip time: 20 minutes
* Most common
    * Day of week: Saturday
    * Pickup zone: LaGuardia Airport (2%)
    * Dropoff zone: Outside of NYC (4%)
    * Pickup borough: Manhattan
    * Dropoff borough: Manhattan
    * Rideshare company: Uber
* Total Driver Pay Box Plot
    * ![summary statistics all fares box plot](Images\pay_box_plot.png)

## Summary Statistics for Top 5% Fares (with round trip in original data)
* Average:
    * Total pay per ride: $69.99
    * Fare: $ 64.79
    * Tip: $ 5.19 (only 35% of customers tip!) 
    * Trip length: 22 miles
    * Trip time: 58 minutes
* Most common
    * Day of week: Thursday
    * Pickup zone: JFK Airport (38%)
    * Dropoff zone: JFK Airport (30%)
    * Pickup borough: Manhattan
    * Dropoff borough: Manhattan
    * Rideshare company: Uber
* Total Driver Pay Box Plot
    * ![summary statistics all fares box plot](Images\top_pay_box_plot.png)

## Data Cleaning Stages (original data)
1. Parquet to CSV (trip_data.csv)
    * .read_parquet (12 files)
    * .sample(1000) each month
    * concat 12 dataframes to 1 dataframe
    * CSV

    * sample size = 12,000

2. General Cleaning for ML (encoded_data.csv)
    1. rename columns
    2. dropping columns to reduce influence/inferences from ml
    3. datetime to .strtime for ml
    4. feature engineering & y-hat engineering
    5. null check (none)
    6. prep for encoding
    7. OneHotEncoder
    8. cat & num to concat to one df
    9. dataframe to csv

## Machine Learning Models For Tips

## Logistic Regression

Logistic Regression was implemented due to its analysis based on dichotomous variables - meaning we determine whether there will be a tip or no tip. For our dataset, tip is represented by 1 and no tip by 0. The goal is to estimate the probability of occurence of drivers receiving tips or not.
Results:

![LRM](Images\LR_Model.png)
![Roc](Images\LR_ROC.png)

**Interpretation**:
The model provided an accuracy score of 99.83, precision and recall of 100% and auc_roc score of 99.84%. The accuracy score determines the accurate assessment of the model. The model, in part for **tips** scored a precision for 100% whereas the **no tip** classifier registered a 100% recall. The issue at hand will be the fact that the precision score does not take into account the `false negative`. A higher recall helps us not to miss out on potential customers who might tip. Due to the imbalance nature of the dataset, we can conclude that the model might have high accuracy score predicting all fares to have received tips without accounting for **no tips**. To resolve this, resampling was implemented to combat this issue.   

### Resampling LR Model with RandomOversampler

![ReLRM](Images\LR_ModelSampled.png)
![Roc](Images\LRO_ROC.png)

With RandomOverSampler solving the problem of imbalanced datasets, tip and no tip `value_counts` had an increase dataset of 7239 each. Results showed an accuracy score of 99.91%, precision and recall all at 100% and auc_roc of 99.92%. With a balance dataset, we can conclude that the accuracy score can be relied on as a better assessment of the model. High Precision and High Recall, help us determine where there will be a tip and take into account we might not miss out on potential customers who will increase our chances of a good tip plus roc score increasing to 99.92. 

For the precaution of making sure that our models are accurate, `XGBoostClassifer` was used to address overfitting while increasing speed and efficiency of our model. XGBoostClassifier resulted to an accuracy score of 100%, high precision and high recall of 100% and roc-auc score of 99.83%. See below for Classification report:

![XGBC](Images\XGBC_Model.png)
![Roc](Images\XGB_ROC.png)

## Random Forest Ensembling Method

Random Forest combines multiple decision trees by averaging to reach a single results which is trianed through Regression and Classification Tree algorithm. Random Forest results will be used to compare the results of Logistic Regression Model.Using the `regular` random forest model - classification report achieved an accuracy score of 99.9161%, precision and recall scores of 100% each accordingly. ROC_AUC curve which measures the classification effectiveness by distingushing between classes scored a 99.92%

![RRF](Images\RRF_Model.png)
![Roc](Images\RF_ROC.png)

A `Balanced` Random Forest Model also achieved a calssification report of an accuracy score of 99.91% , precision and recall scores of 100%. With ROC_AUC score of 99.92%. This can be relied on since the dataset has been balanced between tips and no_tips.

![BRF](Images\BRF_Model.png)
![Roc](Images\RFRe_ROC.png)

Implementing `Smoteenn` as part  of the RandomForestClassifier does a better job over-sampling using the `SMOTE` and cleaning/undersampling using `Edited Nearest Neighbors`. SMOTEENN scored an accuracy of 99.41%, with Precision of 100% and Recall of 99% accordingly. With an ROC_AUC score of 99.41%.

![SMOTEENN](Images\SMOTEENN.png)
![ROC](Images\SMOTEENN_ROC.png)

Based on the Balanced RandomForest and SMOTEENN which balanced out the datasets, it can be concluded that Balanced RandomForest did a better job with an accuracy score of 99.91% , Precision and Recall of 100% , and an roc_auc of 99.92% whereas SMOTEENN scored an accuracy of 99.41%, Precision 100% and Recall of 99% , and roc_curve of 99.41%. To determine which produced the best results, the confusion matrix was analyzed. Balanced RandomForest had TP of 2404 , FP of 0, FN of 1 and TN of 595 ; SMOTEENN produced a TP of 2404 , FP of 0, FN of 7, and TN of 589. It would be best to use the Balanced RandomForest as a benchmark to assess the behavior of extra tipping for drivers due it the models nature of training and learning the dataset to produce 1 false negative. 

## Deep Learning Model - Neural Network
Multi-Layer Perceptron Classifier (MLPC) as a deep learning neural network model trains model implementing backpropagation to master and learn relevant features making sure to implement the Stochastic Gradient Descent (SGD) to asses and respond to the best fit model, and minimizing the loss function in the process. This produced a accuracy score of 96%, Precision of 93% and Recall of 99%, with an roc_auc of 91.13%

![MLPC](Images\MLPC_Model.png)
![ROC](Images\MPC_ROC.png)

##  Machine Learning Conclusion
Taking out the Logistic Regression and the Random Forest models which had imbalanced datasets, all other Machine learning models can be used to and or combined to reach a better ML to implement since every Ml model comes with its different strategies in achieving results. Factoring everything in - confusion matrix, accuracy score, precision and recal, roc_auc score, the best model to use will be LR Oversampled perfromed better with the datasets, followed by Balanced RandomForest, XGBoost Classifier, SMOTEENN and MLPClassifier to decide on the tipping locations for drivers. In the near future, deep learning, MPLClassifier and others could be implemented to train the dataset effectively as I believe that will be a great resource to produce accurate insight and predictions mimicking the human brain in order to help with the aim of the project. 

## Top Fares: 365-Day Prophet Timeseries
#### When are ~5% fares (with return) are more likely?

* Prophet vs. historical data analysis
    * Top ~5% = 1 ride every 3-4 days
    * Prophet is good with missing data, seasonality, inference
* upward trend could be attributed to general economic conditions
* Top Fare Periods:
    * Sun PM – Wed AM
    * Thurs PM – Fri PM
* Top Fare Shifts:
    * AM = 4:00 - 9:00
    * PM = 12:00-19:00
* Are some or all of these surge periods?

* ![prophet plot](Images\prophet_365.png)

* ![prophet y_hat](Images\prophet_y_hat.png)

* ![prophet components](Images\prophet_components.png)

## Driver Schedule Recommendations (based on historical analysis and model output)
* [model implications for tips]
* JFK Airport Runs to Midtown and Lower Manhattan
    * Distances between zones in these areas of Manhattan are small.

* Schedule:
    * M/T/W/F AM = 4:00 - 9:00
    * M/T/Th/F PM = 12:00-19:00

## Issues
* Large Kaggle file --> found original source with smaller files
* Limitations of Github, VS, Colab, Kaggle & handling large files --> limited sample size
* Experimented with Dask & Coiled (not used)


## Topics for further study
* An interactive tool for drivers to evaluate parameters in real time (e.g., drop off location) to maximize earnings
* Utilize more neural networks training it effectively to help with the aim of the project
* Questions as to why people do not tip at all??
* To drop a column or not drop a column? Feature engineering.
* Analyzing airport trip data in more detail
* Surge analysis using pay per mile/minute

## Attributed code, used for adaptation:
* https://github.com/Aweymouth13/rideshare_analysis/blob/main/ETL_analysis.ipynb

        #mapping of column names
        column_mapping = {
            'hvfhs_license_num': 'business',
            'dispatching_base_num': 'd1',
            'originating_base_num': 'd2',
            'request_datetime': 'request_time',
            'on_scene_datetime': 'on_scene_time',
            'pickup_datetime': 'pickup_time',
            'dropoff_datetime': 'dropoff_time',
            'PULocationID': 'pickup_location',
            'DOLocationID': 'dropoff_location',
            'trip_miles': 'trip_length',
            'trip_time': 'trip_time_seconds',
            'base_passenger_fare': 'base_passenger_fare',
            'tolls': 'tolls',
            'bcf': 'bcf',
            'sales_tax': 'sales_tax',
            'congestion_surcharge': 'congestion_surcharge',
            'airport_fee': 'airport_fee',
            'tips': 'tips',
            'driver_pay': 'driver_pay',
            'shared_request_flag': 'd3',
            'shared_match_flag': 'd4',
            'access_a_ride_flag': 'd5',
            'wav_request_flag': 'd6',
            'wav_match_flag': 'd7',
            }

        #columns to drop
        columns_to_drop = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7']


        #rename
        df = df.rename(columns=column_mapping)

        #drop columns
        df = df.drop(columns=columns_to_drop)

