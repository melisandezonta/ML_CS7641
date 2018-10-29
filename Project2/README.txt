{\rtf1\ansi\ansicpg1252\cocoartf1265\cocoasubrtf210
{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
\paperw11900\paperh16840\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural

\f0\fs24 \cf0 Readme : \
The project was made on Java on the Eclipse IDE. The library used was ABAGAIL. On this IDE, only this library was needed.\
\
But all files were processed in R\
\
The packages that need to be installed are : -ggplot2\
								   - readr\
\
\
In order to compute the code, download the code file, 3 files are available : \
\
- ABAGAIL containing : \
Only the src was modified and in the src, only opt.test which contains the classes we created:\
				- ContinuousPeaksProblem.java\
				- KnapsackProblem.java\
				- TravelingSalesmanProblem.java\
				- DiabetesTest.java\
				-diabetes.txt\
				-diabetes_test.csv\
				-diabetes_train.csv\
\
- workspace containing the project file which has :\
2 files of results are created from ABAGAIL : -DiabetesResults from DiabetesTest\
								   - OptmizationResults from the 3 others\
\
It is important to transform the .csv, a first line need to be added like random letters for each column so the first line is not erased when passing through R.\
The images and .csv were left in the files\
- R code  containing : -ContinuousPeaks.R : for plotting the different algorithms\
				 - ContinuousPeaks_parameters.R : for variations of parameters\
				 - Knapsack.R: for plotting the different algorithms\
				 - Knapsack_parameters.R: for variations of parameters\
				 - TSP.R: for plotting the different algorithms\
				 - TSP_parameters.R: for variations of parameters\
				 - diabetes_analysis.R\
				 - diabetes_analysis_bis.R\
\
N.B : - In the opt.test, in the files some comments were put on test of parameters, by uncommenting, the data are created and you can obtain the curves on R.\
-In the R code file, all the .R codes contain path of files in relative because it can cause problem so it would be better to write them entirely.}