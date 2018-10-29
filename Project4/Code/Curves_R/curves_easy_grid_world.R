library("gridExtra")
library('ggplot2')

# Easy Grid World ---------------------------------------------------------

# Steps to Goal---------------------------------------------------------

iterations = seq(1,100)
Value.Iteration.steps = c(565,75,11,11,14,15,9,18,10,9,10,12,13,13,10,10,9,13,10,13,12,11,13,11,12,9,9,16,13,9,11,14,11,9,17,10,10,14,13,9,13,15,9,15,13,10,13,9,10,14,10,18,12,9,13,13,13,9,9,13,12,11,25,13,10,11,10,15,14,9,10,15,13,10,13,13,15,10,9,9,10,11,12,11,11,9,12,10,9,14,9,11,13,9,17,10,9,16,10,10)
Policy.Iteration.steps = c(1434,14,51,12,11,10,14,13,11,12,9,17,11,9,11,11,11,19,13,9,10,17,14,12,9,17,14,11,14,13,12,12,17,17,9,12,13,12,13,13,10,12,10,16,10,9,9,16,9,15,12,10,9,11,11,9,9,13,22,11,10,13,16,16,10,9,10,10,10,10,9,11,9,10,20,14,11,19,11,11,15,11,19,15,10,12,12,10,12,11,13,12,11,17,16,16,11,13,14,9)
Q.Learning.steps = c(69,80,32,40,51,13,29,12,14,11,36,52,108,14,11,66,132,35,13,20,96,52,15,12,12,51,26,10,13,18,28,16,15,21,16,13,11,85,12,30,30,138,9,11,20,15,11,24,27,11,38,30,9,11,23,41,16,12,24,20,15,11,12,49,16,17,11,17,10,11,9,10,10,9,14,77,19,13,10,13,9,13,76,12,21,50,16,10,11,26,21,9,49,15,25,9,13,22,16,18)

easy.grid.world.value.iteration.steps = data.frame(x = iterations, y = Value.Iteration.steps)
easy.grid.world.policy.iteration.steps = data.frame(x = iterations, y = Policy.Iteration.steps)
easy.grid.world.Q.Learning.iteration.steps = data.frame(x = iterations, y = Q.Learning.steps)
easy.grid.world.steps = data.frame(x = c(iterations,iterations,iterations),
                                   y = c(Value.Iteration.steps,Policy.Iteration.steps,Q.Learning.steps),
                                   Algorithm = c(rep("Value Iteration",length(iterations)),rep("Policy Iteration",length(iterations)),rep("Q Learning",length(iterations))))

p1 <- ggplot(easy.grid.world.value.iteration.steps,aes(x=x,y=y))+geom_line(color = 'orange')+
  ggtitle('Easy Grid World (Value Iteration)')+
  xlab('Number of Iterations') + ylab('Number of Steps to Goal') + scale_y_log10()
p2 <- ggplot(easy.grid.world.policy.iteration.steps,aes(x=x,y=y))+geom_line(color = 'green')+
  ggtitle('Easy Grid World (Policy Iteration)')+
  xlab('Number of Iterations') + ylab('Number of Steps to Goal')+ scale_y_log10()
p3 <- ggplot(easy.grid.world.Q.Learning.iteration.steps,aes(x=x,y=y))+geom_line(color = 'red')+
ggtitle('Easy Grid World (Q Learning)')+xlab('Number of Iterations') + ylab('Number of Steps to Goal')
p4 <- ggplot(easy.grid.world.steps,aes(x=x,y=y,color = Algorithm))+geom_line()+ggtitle('Easy Grid World (Steps to Goal)')+xlab('Number of Iterations') +
 ylab('Number of Steps to Goal') + ylim(c(0,600))

p <- grid.arrange(p1, p2, p3, p4, ncol=2,nrow = 2)
#ggsave("easy_grid_world_steps_to_goal.png",width = 10,height = 10)


# Time to calculate Policy---------------------------------------------------------

iterations = seq(1,100)
Value.Iteration.time = c(112,3,3,6,16,5,5,5,6,6,6,6,6,29,9,13,42,15,8,20,10,11,10,18,43,24,12,14,17,30,34,33,50,13,12,13,15,14,15,38,18,15,22,15,28,20,33,35,36,32,31,44,28,24,18,20,18,28,42,36,39,41,33,29,54,42,27,44,34,53,62,55,83,61,94,96,59,33,38,43,32,52,52,36,58,40,47,40,83,72,38,40,93,38,66,44,49,57,108,96)
Policy.Iteration.time = c(9,5,15,15,24,12,17,16,13,14,44,17,20,57,23,56,33,50,59,80,19,19,32,57,46,51,61,30,55,65,48,28,28,29,29,33,34,50,49,46,37,50,45,103,59,38,39,63,98,101,53,49,51,100,53,97,105,131,163,244,209,116,61,81,76,117,88,79,124,120,90,103,105,143,137,105,130,93,95,73,114,77,114,123,111,70,85,96,117,126,58,71,56,97,106,92,81,67,91,62)
Q.Learning.time = c(49,8,7,3,6,4,2,7,7,6,28,10,8,19,6,5,9,3,4,8,8,10,5,6,5,10,22,6,6,10,6,6,6,15,10,14,6,14,8,15,10,29,20,35,11,7,10,9,20,48,7,11,12,12,13,12,11,9,16,35,27,8,15,16,11,22,13,9,11,10,14,13,9,15,9,9,15,10,10,12,13,22,11,15,15,16,6,8,9,17,16,11,11,13,6,9,10,15,13,13)

easy.grid.world.value.time = data.frame(x = iterations, y = Value.Iteration.time)
easy.grid.world.policy.time = data.frame(x = iterations, y = Policy.Iteration.time )
easy.grid.world.Q.Learning.time = data.frame(x = iterations, y = Q.Learning.time)
easy.grid.world.time = data.frame(x = c(iterations,iterations,iterations),
                                   y = c(Value.Iteration.time,Policy.Iteration.time,Q.Learning.time),
                                   Algorithm = c(rep("Value Iteration",length(iterations)),rep("Policy Iteration",length(iterations)),rep("Q Learning",length(iterations))))

p5 <- ggplot(easy.grid.world.value.time,aes(x=x,y=y))+geom_line(color = 'orange')+
  ggtitle('Easy Grid World (Value Iteration)')+
  xlab('Number of Iterations') + ylab('Number of Milliseconds to Calculate Policy')
p6 <- ggplot(easy.grid.world.policy.time,aes(x=x,y=y))+geom_line(color = 'green')+
  ggtitle('Easy Grid World (Policy Iteration)')+
  xlab('Number of Iterations') + ylab('Number of Milliseconds to Calculate Policy')
p7 <- ggplot(easy.grid.world.Q.Learning.time,aes(x=x,y=y))+geom_line(color = 'red')+
  ggtitle('Easy Grid World (Q Learning)')+xlab('Number of Iterations') + ylab('Number of Milliseconds to Calculate Policy')
p8 <- ggplot(easy.grid.world.time,aes(x=x,y=y,color = Algorithm))+geom_line()+ggtitle('Easy Grid World (Time to calculate Policy)')+xlab('Number of Iterations') +
  ylab('Number of Milliseconds to Calculate Policy') 

grid.arrange(p5, p6, p7, p8, ncol=2,nrow = 2)
#ggsave("easy_grid_world_steps_to_goal.png",width = 10,height = 10)

# Rewards---------------------------------------------------------

iterations = seq(1,100)
Value.Iteration.rewards = c(-463.0,27.0,91.0,91.0,88.0,87.0,93.0,84.0,92.0,93.0,92.0,90.0,89.0,89.0,92.0,92.0,93.0,89.0,92.0,89.0,90.0,91.0,89.0,91.0,90.0,93.0,93.0,86.0,89.0,93.0,91.0,88.0,91.0,93.0,85.0,92.0,92.0,88.0,89.0,93.0,89.0,87.0,93.0,87.0,89.0,92.0,89.0,93.0,92.0,88.0,92.0,84.0,90.0,93.0,89.0,89.0,89.0,93.0,93.0,89.0,90.0,91.0,77.0,89.0,92.0,91.0,92.0,87.0,88.0,93.0,92.0,87.0,89.0,92.0,89.0,89.0,87.0,92.0,93.0,93.0,92.0,91.0,90.0,91.0,91.0,93.0,90.0,92.0,93.0,88.0,93.0,91.0,89.0,93.0,85.0,92.0,93.0,86.0,92.0,92.0)
Policy.Iteration.rewards = c(-1332.0,88.0,51.0,90.0,91.0,92.0,88.0,89.0,91.0,90.0,93.0,85.0,91.0,93.0,91.0,91.0,91.0,83.0,89.0,93.0,92.0,85.0,88.0,90.0,93.0,85.0,88.0,91.0,88.0,89.0,90.0,90.0,85.0,85.0,93.0,90.0,89.0,90.0,89.0,89.0,92.0,90.0,92.0,86.0,92.0,93.0,93.0,86.0,93.0,87.0,90.0,92.0,93.0,91.0,91.0,93.0,93.0,89.0,80.0,91.0,92.0,89.0,86.0,86.0,92.0,93.0,92.0,92.0,92.0,92.0,93.0,91.0,93.0,92.0,82.0,88.0,91.0,83.0,91.0,91.0,87.0,91.0,83.0,87.0,92.0,90.0,90.0,92.0,90.0,91.0,89.0,90.0,91.0,85.0,86.0,86.0,91.0,89.0,88.0,93.0)
Q.Learning.rewards = c(33.0,22.0,70.0,62.0,51.0,89.0,73.0,90.0,88.0,91.0,66.0,50.0,-6.0,88.0,91.0,36.0,-30.0,67.0,89.0,82.0,6.0,50.0,87.0,90.0,90.0,51.0,76.0,92.0,89.0,84.0,74.0,86.0,87.0,81.0,86.0,89.0,91.0,17.0,90.0,72.0,72.0,-36.0,93.0,91.0,82.0,87.0,91.0,78.0,75.0,91.0,64.0,72.0,93.0,91.0,79.0,61.0,86.0,90.0,78.0,82.0,87.0,91.0,90.0,53.0,86.0,85.0,91.0,85.0,92.0,91.0,93.0,92.0,92.0,93.0,88.0,25.0,83.0,89.0,92.0,89.0,93.0,89.0,26.0,90.0,81.0,52.0,86.0,92.0,91.0,76.0,81.0,93.0,53.0,87.0,77.0,93.0,89.0,80.0,86.0,84.0)

easy.grid.world.value.rewards = data.frame(x = iterations, y = Value.Iteration.rewards)
easy.grid.world.policy.rewards  = data.frame(x = iterations, y = Policy.Iteration.rewards)
easy.grid.world.Q.Learning.rewards  = data.frame(x = iterations, y = Q.Learning.rewards)
easy.grid.world.steps = data.frame(x = c(iterations,iterations,iterations),
                                   y = c(Value.Iteration.rewards,Policy.Iteration.rewards ,Q.Learning.rewards),
                                   Algorithm = c(rep("Value Iteration",length(iterations)),rep("Policy Iteration",length(iterations)),rep("Q Learning",length(iterations))))

p9 <- ggplot(easy.grid.world.value.rewards,aes(x=x,y=y))+geom_line(color = 'orange')+
  ggtitle('Easy Grid World (Value Iteration)')+
  xlab('Number of Iterations') + ylab('Number of rewards gained for the optimal policy') + ylim(c(-100,max(Value.Iteration.rewards)))
p10 <- ggplot(easy.grid.world.policy.rewards,aes(x=x,y=y))+geom_line(color = 'green')+
  ggtitle('Easy Grid World (Policy Iteration)')+
  xlab('Number of Iterations') + ylab('Number of rewards gained for the optimal policy') + ylim(c(0,max(Policy.Iteration.rewards)))
p11 <- ggplot(easy.grid.world.Q.Learning.rewards,aes(x=x,y=y))+geom_line(color = 'red')+
  ggtitle('Easy Grid World (Q Learning)')+xlab('Number of Iterations') + ylab('Number of rewards gained for the optimal policy')
p12 <- ggplot(easy.grid.world.steps,aes(x=x,y=y,color = Algorithm))+geom_line()+ggtitle('Easy Grid World (total reward)')+xlab('Number of Iterations') +
  ylab('Number of rewards gained for the optimal policy')

grid.arrange(p9, p10, p11, p12, ncol=2,nrow = 2)
#ggsave("easy_grid_world_steps_to_goal.png",width = 10,height = 10)

