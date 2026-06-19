# PROBLEM STATEMENT

Cities worldwide are increasingly turning to AI-powered solutions to tackle the issue of traffic congestion, which disrupts the smooth flow of transportation and poses a significant barrier to economic growth. To address this challenge effectively, the first step is to comprehensively understand travel demand and patterns within urban areas.

# The Task

Design a system that provides valuable insights into passenger travel patterns, booking behavior, and trip cancellations. These insights will be used for various analyses and to predict demand in the travel industry, enabling cities to implement data-driven strategies to alleviate traffic congestion and promote more efficient mobility.


# Dataset Description
The dataset folder contains the following files:

train.csv: 77,299 rows × 11 columns

test.csv: 41,778 rows × 10 columns

sample_submission.csv: 5 rows × 2 columns


# Variable Description
The columns provided in the dataset are defined as follows:
| Column Name | Description |
| :--- | :--- |
| **Index** | Represents the unique identification of datapoint |
| **geohash** | Represents geographic information regarding a place |
| **day** | Represents the day when the information is recorded |
| **timestamp** | Represents the timestamp of the record inserted in the system |
| **RoadType** | Represents the type of road in the nearby location |
| **NumberofLanes** | Represents the number of roads/lanes present in the location |
| **LargeVehicles** | Represents whether large vehicles are permitted on the specific roads/lanes |
| **Landmarks** | Represents whether there are any landmarks near the location |
| **Temperature** | Represents the temperature of the place |
| **Weather** | Represents the weather of the place |
| **demand** | Represents the demand of the traffic at the timestamp |


# Evaluation Metric
The performance of the predictions will be evaluated using the $R^2$ score, formatted as:$$\text{Score} = \max(0, 100 \times \text{r2\_score}(\text{actual}, \text{predicted}))$$