df.sa.temp = read.csv("./OptimizationResults/ContinuousPeaksSA_Temperature.csv")
names(df.sa.temp)=c("iteration","fitness","time","temperature","cooling.factor")
df.sa.temp.median = aggregate(df.sa.temp, list(iteration = df.sa.temp$iteration,
                                     initial_temperature = df.sa.temp$temperature,
                                     cooling_factor = df.sa.temp$cooling.factor), FUN = median)


df.sa.temp.median$temperature=as.character(df.sa.temp.median$temperature)
ggplot(df.sa.temp.median,aes(iteration,fitness,col=temperature)) + geom_line() +xlim(c(1000,5000))+ylim(c(200,300))+
  ggtitle('Fitness values over different temperature')+
  xlab('iterations')+
  ylab('Fitness Values')
ggplot(df.sa.temp.median,aes(iteration,time,col=temperature)) + geom_point() + geom_line()

df.sa.cooling = read.csv("./OptimizationResults/ContinuousPeaksSA_cooling.csv")
names(df.sa.cooling)=c("iteration","fitness","time","temperature","cooling.factor")
df.sa.cooling.median = aggregate(df.sa.cooling, list(iteration = df.sa.cooling$iteration,
                                               initial_temperature = df.sa.cooling$cooling.factor,
                                               cooling_factor = df.sa.cooling$cooling.factor), FUN = median)


df.sa.cooling.median$cooling.factor=as.character(df.sa.cooling.median$cooling.factor)
ggplot(df.sa.cooling.median,aes(iteration,fitness,col=cooling.factor)) + geom_line()+
  ggtitle('Fitness values over different values of cooling factor')+
  xlab('iterations')+
  ylab('Fitness Values')
  
ggplot(df.sa.cooling.median,aes(iteration,time,col=cooling.factor)) + geom_point()+geom_line()
