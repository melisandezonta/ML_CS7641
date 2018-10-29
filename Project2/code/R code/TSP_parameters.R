library(ggplot2)
library(readr)
df.pop = read_csv("~/Documents/GTL_courses/Machine_Learning/Homework2/workspace/project/OptimizationResults/tspGA_POP.csv")
names(df.pop)=c("iteration","fitness","time","population_size","toMate","toMutate")
df.pop.max = aggregate(df.pop, list(iteration = df.pop$iteration,
                                               toMate=df.pop$toMate,
                                               toMutate = df.pop$toMutate), FUN = mean)


ggplot(df.pop.max,aes(toMate,toMutate)) + geom_point(aes(col=df.pop.max$fitness))

ggplot(df.pop.max,aes(toMate,toMutate)) + geom_point(aes(col=df.pop.max$time)) 


