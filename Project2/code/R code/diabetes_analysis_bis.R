library(ggplot2)
## Curves of Comparaison between algos erreur/trainingTime/Percentage_Test in function des iterations
# Diabetes Problem RHC
library(readr)


df.rhc = read_csv("./DiabetesResults/DiabetesRHC_best.csv")
names(df.rhc)=c("iteration","run","error","percentage_training","trainingTime","correct_instances","incorrect_instances","percentage_test")

subset_df_rhc_training = data.frame("iteration" = df.rhc$iteration,"percentage"= df.rhc$percentage_training)
subset_df_rhc_test = data.frame("iteration" = df.rhc$iteration,"percentage" = df.rhc$percentage_test)

subset_df.rhc =rbind(subset_df_rhc_training,subset_df_rhc_test)

a=c(rep("training.accuracy",100),rep("testing.accuracy",100))

subset_df.rhc$Accuracy = a 

ggplot(subset_df.rhc,aes(x=iteration,y=percentage,col = Accuracy))+geom_smooth()+
  ggtitle('Learning and Testing Curves for Neural Network RHC')+
  xlab('Iterations') + 
  ylab('Correctly classified (%)')


# Diabetes  Problem SA
#10^6
df.sa = read.csv("./DiabetesResults/DiabetesSA_IT1000000.0_CF0.95.csv")
names(df.sa)=c("iteration","run","error","percentage_training","trainingTime","correct_instances","incorrect_instances","percentage_test","Initial.Temperature","Cooling.Factor")

subset_df_sa_training = data.frame("iteration" = df.sa$iteration,"percentage"= df.sa$percentage_training)
subset_df_sa_test = data.frame("iteration" = df.sa$iteration,"percentage" = df.sa$percentage_test)

subset_df.sa =rbind(subset_df_sa_training,subset_df_sa_test)

a=c(rep("training.accuracy",100),rep("testing.accuracy",100))

subset_df.sa$Accuracy = a 

ggplot(subset_df.sa,aes(x=iteration,y=percentage,col = Accuracy))+geom_smooth()+
  ggtitle('Learning and Testing Curves for Neural Network SA')+
  xlab('Iterations') + 
  ylab('Correctly classified (%)')


# Map Problem SA

df.pop.sa = read.csv("./DiabetesResults/DiabetesSA_IT.csv")
names(df.pop.sa)=c("iteration","run","error","percentage_training","trainingTime","correct_instances","incorrect_instances","percentage_test","Cooling.Factor","Initial.Temperature")
df.pop.mean = aggregate(df.pop.sa, list(iteration = df.pop.sa$iteration,
                                    CE =df.pop.sa$Cooling.Factor,
                                    IT = df.pop.sa$Initial.Temperature), FUN = mean)

training.accuracy = df.pop.mean$percentage_training
testing.accuracy = df.pop.mean$percentage_test
ggplot(df.pop.mean,aes(CE,log10(df.pop.mean$Initial.Temperature))) + geom_point(aes(col=testing.accuracy))+
  xlab('Cooling Factor') + 
  ylab('Initial Temperature')

ggplot(df.pop.mean,aes(CE,log10(df.pop.mean$Initial.Temperature))) + geom_point(aes(col=training.accuracy))+
  xlab('Cooling Factor') + 
  ylab('Initial Temperature')


# Diabetes Problem GA

df.ga = read.csv("./DiabetesResults/DiabetesGA_POP500_MAT100_MUT40_old.csv")
names(df.ga)=c("iteration","run","error","percentage_training","trainingTime","correct_instances","incorrect_instances","percentage_test","population.Size","toMate","toMutate")

subset_df_ga_training = data.frame("iteration" = df.ga$iteration,"percentage"= df.ga$percentage_training)
subset_df_ga_test = data.frame("iteration" = df.ga$iteration,"percentage" = df.ga$percentage_test)

subset_df.ga =rbind(subset_df_ga_training,subset_df_ga_test)

a=c(rep("training.accuracy",100),rep("testing.accuracy",100))

subset_df.ga$Accuracy = a 

ggplot(subset_df.ga,aes(x=iteration,y=percentage,col = Accuracy))+geom_smooth()+
  ggtitle('Learning and Testing Curves for Neural Network GA')+
  xlab('Iterations') + 
  ylab('Correctly classified (%)')

# Variation of parameters in GA

df.ga_test = read_csv("./DiabetesResults/DiabetesGA_POP.csv")
names(df.ga_test)=c("iteration","run","error","percentage_training","trainingTime","correct_instances","incorrect_instances","percentage_test","population.Size","toMate","toMutate")

df.ga_test.median = aggregate(df.ga_test, list(iteration = df.ga_test$iteration,toMutate=df.ga_test$toMutate), FUN = median)
df.ga_test.median$toMutate=as.character(df.ga_test.median$toMutate)

ggplot(df.ga_test.median,aes(iteration,percentage_training,col=toMutate))+geom_line()+
  ggtitle('Testing Curve for different parameters')+
  xlab('Iterations') + 
  ylab('Correctly classified (%)')

# Diabetes  Problem ANN
df.ANN = read.csv("./DiabetesResults/NeuralNetworkBP_best.csv")
names(df.ANN)=c("iteration","run","error","percentage_training","trainingTime","correct_instances","incorrect_instances","percentage_test")


subset_df_ANN_training = data.frame("iteration" = df.ANN$iteration,"percentage"= df.ANN$percentage_training)
subset_df_ANN_test = data.frame("iteration" = df.ANN$iteration,"percentage" = df.ANN$percentage_test)

subset_df.ANN =rbind(subset_df_ANN_training,subset_df_ANN_test)

a=c(rep("training.accuracy",100),rep("testing.accuracy",100))

subset_df.ANN$Accuracy = a 

ggplot(subset_df.ANN,aes(x=iteration,y=percentage,col = Accuracy))+geom_smooth()+
  ggtitle('Learning and Testing Curves for Neural Network ANN')+
  xlab('Iterations') + 
  ylab('Correctly classified (%)')

