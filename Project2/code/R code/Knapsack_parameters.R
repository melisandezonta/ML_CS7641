library(ggplot2)
library(readr)
df.samples = read.csv("./OptimizationResults/KnapsackMIMIC_SAMPLES.csv")

df.samples.median = aggregate(df.samples, list(iteration = df.samples$iteration,
                                               samples=df.samples$samples,
                                               tokeep = df.samples$tokeep), FUN = median)

df.samples.median$samples=as.character(df.samples.median$samples)

ggplot(df.samples.median,aes(iteration,fitness,col=samples)) + geom_point() + geom_line()+
  ggtitle('Fitness values over different values of samples')+
  xlab('iterations')+
  ylab('Fitness Values')

ggplot(df.samples.median,aes(iteration,time)) + geom_point() + geom_line()


df.tokeep = read.csv("./OptimizationResults/KnapsackMIMIC_tokeep.csv")
names(df.tokeep)=c("iteration","fitness","time","samples","tokeep")
df.tokeep.median = aggregate(df.tokeep, list(iteration = df.tokeep$iteration,
                                               samples=df.tokeep$samples,
                                               tokeep = df.tokeep$tokeep), FUN = median)

df.tokeep.median$samplesr=as.character(df.tokeep.median$tokeep)

ggplot(df.tokeep.median,aes(iteration,fitness,col=tokeep)) + geom_point() + geom_line()+
  ggtitle('Fitness values over different values of number of elements to keep')+
  xlab('iterations')+
  ylab('Fitness Values')
ggplot(df.tokeep.median,aes(iteration,time,col=tokeep)) + geom_point() + geom_line()
