library(ggplot2)

# Knapsack Problem RHC
library(readr)


df.rhc = read_csv("./OptimizationResults/KnapsackRHC.csv")
names(df.rhc)=c("iteration","fitness","time")
df.rhc.median = aggregate(df.rhc, list(iteration = df.rhc$iteration), FUN = median)


p1=ggplot(df.rhc.median,aes(iteration,fitness)) + geom_point() + geom_line()
ggplot(df.rhc.median,aes(iteration,time)) + geom_point() + geom_line()

# Knapsack Problem SA

df.sa = read.csv("./OptimizationResults/KnapsackSA_IT100.0_CF0.95.csv")
names(df.sa)=c("iteration","fitness","time","temperature","cooling.factor")
df.sa.median = aggregate(df.sa, list(iteration = df.sa$iteration,
                                     initial_temperature = df.sa$temperature,
                                     cooling_factor = df.sa$cooling.factor), FUN = median)

p2=ggplot(df.sa.median,aes(iteration,fitness)) + geom_point() + geom_line()
ggplot(df.sa.median,aes(iteration,time)) + geom_point() + geom_line()

# Knapsack Problem GA

df.ga = read.csv("./OptimizationResults/KnapsackGA_POP200_MAT150_MUT25.csv")
names(df.ga)=c("iteration","fitness","time","population_size","toMate","toMutate")
df.ga.median =aggregate(df.ga, list(iteration = df.ga$iteration,
                                    population_size = df.ga$population_size,
                                    toMate = df.ga$toMate,
                                    toMutate=df.ga$toMutate
                                    ), FUN = median)

p3=ggplot(df.ga.median,aes(iteration,fitness)) + geom_point() + geom_line()
ggplot(df.ga.median,aes(iteration,time)) + geom_point() + geom_line()

# Knapsack Problem MIMIC
df.mimic = read.csv("./OptimizationResults/KnapsackMIMIC.csv")
names(df.mimic)=c("iteration","fitness","time","samples","tokeep")
df.mimic.median = aggregate(df.mimic, list(iteration = df.mimic$iteration,
                                     samples = df.mimic$samples,
                                     tokeep = df.mimic$tokeep), FUN = median)

p4=ggplot(df.mimic.median,aes(iteration,fitness)) + geom_point() + geom_line()
ggplot(df.mimic.median,aes(iteration,time)) + geom_point() + geom_line()

df.rhc.median=df.rhc.median[c("iteration","fitness","time")]
df.sa.median=df.sa.median[names(df.rhc)]
df.ga.median=df.ga.median[names(df.rhc)]
df.mimic.median=df.mimic.median[c("iteration","fitness","time")]

df=rbind(df.rhc.median,df.sa.median,df.ga.median,df.mimic.median)

a=c(rep("rhc",500),rep("sa",500),rep("ga",500),rep("mimic",100))
df$algorithm=a

ggplot(df,aes(x=iteration,y=fitness,col=algorithm))+geom_line()+
  ggtitle('Fitness values over iterations')+
  xlab('iterations')+
  ylab('Fitness Values')

ggplot(df,aes(x=iteration,y=time,col=algorithm))+geom_line()+
  ggtitle('Time over iterations')+
  xlab('iterations')+
  ylab('Time in s')

