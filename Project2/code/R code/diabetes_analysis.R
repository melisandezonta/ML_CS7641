library(ggplot2)
## Courbes of Comparaison between algos error/trainingTime/Percentage_Test in function of iterations
# Diabetes Problem RHC
library(readr)


df.rhc = read_csv(".t/DiabetesResults/DiabetesRHC_best.csv")
names(df.rhc)=c("iteration","run","error","pecentage_training","trainingTime","correct_instances","incorrect_instances","percentage_test")
subset_df.rhc = subset(df.rhc, df.rhc$run == 5)

p1=ggplot(subset_df.rhc,aes(iteration,percentage_test)) + geom_point() + geom_line()
p1_a = ggplot(subset_df.rhc,aes(iteration,trainingTime)) + geom_point() + geom_line()
p1_b = ggplot(subset_df.rhc,aes(iteration,error)) + geom_point() + geom_line()

# Diabetes  Problem SA

df.sa = read.csv("./DiabetesResults/DiabetesSA_IT100.0_CF0.95.csv")
names(df.sa)=c("iteration","run","error","pecentage_training","trainingTime","correct_instances","incorrect_instances","percentage_test","Initial.Temperature","Cooling.Factor")
subset_df.sa = subset(df.sa, df.sa$run == 8)

p2=ggplot(subset_df.sa,aes(iteration,percentage_test)) + geom_point() + geom_line()
p2_a = ggplot(subset_df.sa,aes(iteration,trainingTime)) + geom_point() + geom_line()
p2_b = ggplot(subset_df.sa,aes(iteration,error)) + geom_point() + geom_line()

# Diabetes Problem GA

df.ga = read.csv("./DiabetesResults/DiabetesGA_POP500_MAT100_MUT40_old.csv")
names(df.ga)=c("iteration","run","error","pecentage_training","trainingTime","correct_instances","incorrect_instances","percentage_test","population.Size","toMate","toMutate")
subset_df.ga = subset(df.ga, df.ga$run == 2)

p3=ggplot(subset_df.ga,aes(iteration,percentage_test)) + geom_point() + geom_line()
p3_a = ggplot(subset_df.ga,aes(iteration,trainingTime)) + geom_point() + geom_line()
p3_b = ggplot(subset_df.ga,aes(iteration,error)) + geom_point() + geom_line()

# Diabetes  Problem ANN
df.ANN = read.csv("./DiabetesResults/NeuralNetworkBP.csv")
names(df.ANN)=c("iteration","run","error","pecentage_training","trainingTime","correct_instances","incorrect_instances","percentage_test")
subset_df.ANN = subset(df.ANN, df.ANN$run == 5)

p4=ggplot(subset_df.ANN,aes(iteration,percentage_test)) + geom_point() + geom_line()
p4_a = ggplot(subset_df.ANN,aes(iteration,trainingTime)) + geom_point() + geom_line()
p4_b = ggplot(subset_df.ANN,aes(iteration,error)) + geom_point() + geom_line()

subset_df.rhc=subset_df.rhc[names(df.rhc)]
subset_df.sa=subset_df.sa[names(df.rhc)]
subset_df.ga=subset_df.ga[names(df.rhc)]
subset_df.ANN=subset_df.ANN[names(df.rhc)]

subset_df=rbind(subset_df.rhc,subset_df.sa,subset_df.ga,subset_df.ANN)

a=c(rep("rhc",100),rep("sa",100),rep("ga",100),rep("ANN",100))
subset_df$algorithm=a

ggplot(subset_df,aes(x=iteration,y=percentage_test,col=algorithm))+geom_line()+ggtitle('Testing Accuracy over iterations')+xlab('Iterations') + ylab('Correctly classified (%)')
ggplot(subset_df,aes(x=iteration,y=trainingTime,col=algorithm))+geom_line()+ggtitle('Training time over iterations')+xlab('Iterations') + ylab('Time in s')
subset_df=rbind(subset_df.rhc,subset_df.sa,subset_df.ga)
a=c(rep("rhc",100),rep("sa",100),rep("ga",100))
subset_df$algorithm=a
ggplot(subset_df,aes(x=iteration,y=error,col=algorithm))+geom_line()+ggtitle('Sum of squarred error over iterations')+xlab('Iterations') + ylab('Error')
